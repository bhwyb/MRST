import numpy as np
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc
from dolfinx.mesh import CellType
import basix
import ufl

class MultiphaseFlowSimulator:
    def __init__(self, nx=30, ny=30, nz=3, **kwargs):
        """初始化多相流模拟器
        
        参数:
            nx, ny, nz: 网格数量
            kwargs: 可选参数
                - Lx, Ly, Lz: 区域尺寸 (m)
                - k: 渗透率 (m²)
                - phi: 孔隙度
                - mu_w: 水相粘度 (Pa·s)
                - mu_o: 油相粘度 (Pa·s)
                - Sw_init: 初始水饱和度
                - dt: 时间步长 (s)
        """
        # MPI通信器
        self.comm = MPI.COMM_WORLD
        
        # 网格参数
        self.Lx = kwargs.get('Lx', 100.0)
        self.Ly = kwargs.get('Ly', 100.0)
        self.Lz = kwargs.get('Lz', 20.0)
        
        # 创建网格
        self.mesh = mesh.create_box(
            self.comm,
            [np.array([0.0, 0.0, 0.0]), np.array([self.Lx, self.Ly, self.Lz])],
            [nx, ny, nz]
        )
        
        # 创建函数空间
        # 速度空间（向量空间）
        RT = ("Raviart-Thomas", 1)
        self.RT = dolfinx.fem.functionspace(self.mesh, RT)
        
        # 压力空间（标量空间）
        P1 = ("Lagrange", 1)
        self.P1 = dolfinx.fem.functionspace(self.mesh, P1)
        
        # 创建混合函数空间
        cell = self.mesh.basix_cell()
        RT_element = basix.ufl.element("Raviart-Thomas", cell, 1)
        P1_element = basix.ufl.element("Lagrange", cell, 1)
        mixed_element = basix.ufl.mixed_element([RT_element, P1_element])
        self.W = dolfinx.fem.functionspace(self.mesh, mixed_element)
        
        # 岩石和流体属性
        self.k = fem.Constant(self.mesh, PETSc.ScalarType(kwargs.get('k', 200e-13)))
        self.phi = fem.Constant(self.mesh, PETSc.ScalarType(kwargs.get('phi', 0.25)))
        self.mu_w = fem.Constant(self.mesh, PETSc.ScalarType(kwargs.get('mu_w', 1e-3)))
        self.mu_o = fem.Constant(self.mesh, PETSc.ScalarType(kwargs.get('mu_o', 5e-3)))
        
        # 初始化饱和度
        self.Sw = fem.Function(self.P1)
        self.Sw.interpolate(lambda x: np.full_like(x[0], kwargs.get('Sw_init', 0.8)))
        
        # 时间步长
        self.dt = kwargs.get('dt', 43200)
        
        # 保存结果
        self.pressure_history = []
        self.saturation_history = []
        self.time_steps = []
        
        # 创建results文件夹
        if self.comm.rank == 0:
            if not os.path.exists('results'):
                os.makedirs('results')
    
    def rel_perm(self, Sw):
        """计算相对渗透率"""
        # 创建相对渗透率函数
        krw = fem.Function(self.P1)
        kro = fem.Function(self.P1)
        
        # 计算相对渗透率值（添加小的正数避免零移动度）
        eps = 1e-6  # 小的正数
        Sw_array = np.clip(Sw.x.array, eps, 1.0-eps)  # 限制饱和度范围
        krw.x.array[:] = Sw_array * Sw_array + eps
        kro.x.array[:] = (1.0 - Sw_array) * (1.0 - Sw_array) + eps
        
        return krw, kro
    
    def solve_pressure(self):
        """求解压力方程"""
        # 创建试验函数和测试函数
        w = ufl.TestFunction(self.W)
        u = ufl.TrialFunction(self.W)
        (v, q) = ufl.split(w)
        (u_trial, p_trial) = ufl.split(u)
        
        # 计算总移动度
        krw, kro = self.rel_perm(self.Sw)
        lambda_t = fem.Function(self.P1)
        lambda_t.x.array[:] = krw.x.array/self.mu_w.value + kro.x.array/self.mu_o.value
        
        # 定义混合形式方程
        dx = ufl.dx
        k_const = fem.Constant(self.mesh, PETSc.ScalarType(self.k.value))
        a = (ufl.inner(v, u_trial)/(k_const * lambda_t) + ufl.div(v)*p_trial + ufl.div(u_trial)*q)*dx
        
        # 定义源汇项
        f = fem.Function(self.P1)
        self.setup_wells(f)
        L = f*q*dx
        
        # 创建解函数
        w_solution = fem.Function(self.W)
        
        # 设置边界条件
        # 在外边界设置零流量条件
        facets = mesh.locate_entities_boundary(
            self.mesh, 2,
            lambda x: np.full_like(x[0], True, dtype=bool))
        
        # 获取速度子空间并collapse
        velocity_space = self.RT
        dofs_v = fem.locate_dofs_topological(velocity_space, 2, facets)
        
        # 创建零速度边界条件
        u_bc_func = fem.Function(velocity_space)
        u_bc_func.x.array[:] = 0.0
        bc = fem.dirichletbc(u_bc_func, dofs_v)
        
        # 固定一个压力点以确保唯一解
        vertices = mesh.locate_entities_boundary(
            self.mesh, 0,
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0))
        
        # 获取压力子空间
        pressure_space = self.P1
        dofs_p = fem.locate_dofs_topological(pressure_space, 0, vertices)
        
        # 创建零压力边界条件
        p_bc_func = fem.Function(pressure_space)
        p_bc_func.x.array[:] = 0.0
        bc_pressure = fem.dirichletbc(p_bc_func, dofs_p)
        
        bcs = [bc, bc_pressure]
        
        # 组装系统
        A = petsc.create_matrix(fem.form(a))
        b = petsc.create_vector(fem.form(L))
        
        # 组装矩阵和向量
        petsc.assemble_matrix(A, fem.form(a), bcs=bcs)
        A.assemble()
        petsc.assemble_vector(b, fem.form(L))
        petsc.apply_lifting(b, [fem.form(a)], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # 设置求解器
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        
        # 使用GMRES求解器配合LU预处理
        ksp.setType("gmres")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        
        # 设置求解器参数
        ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
        ksp.setMonitor(lambda ksp, it, rnorm: print(f"迭代 {it}, 残差范数: {rnorm}") if self.comm.rank == 0 else None)
        
        # 求解系统
        x = A.createVecRight()
        ksp.solve(b, x)
        
        # 将解复制到函数中
        w_solution.x.array[:] = x.array[:]
        w_solution.x.scatter_forward()
        
        # 创建子函数来提取速度和压力
        velocity = fem.Function(self.RT)
        pressure = fem.Function(self.P1)
        
        # 获取混合解的子函数
        u_sub = w_solution.sub(0)
        p_sub = w_solution.sub(1)
        
        # 使用interpolate方法复制解
        velocity.interpolate(u_sub)
        pressure.interpolate(p_sub)
        
        return velocity, pressure
    
    def solve_transport(self, velocity):
        """求解输运方程"""
        # 创建试验函数和测试函数
        Sw_new = ufl.TrialFunction(self.P1)
        v = ufl.TestFunction(self.P1)
        
        # 计算分流函数
        krw, kro = self.rel_perm(self.Sw)
        fw = fem.Function(self.P1)
        fw.x.array[:] = (krw.x.array/self.mu_w.value) / (krw.x.array/self.mu_w.value + kro.x.array/self.mu_o.value)
        
        # 定义方程
        dt = fem.Constant(self.mesh, PETSc.ScalarType(self.dt))
        dx = ufl.dx
        F = (Sw_new - self.Sw)/dt*v*dx + ufl.dot(velocity, ufl.grad(fw))*v*dx
        
        # 求解
        Sw_solution = fem.Function(self.P1)
        
        # 设置边界条件（入口处的饱和度）
        def inflow_boundary(x):
            return np.isclose(x[0], self.Lx) & np.isclose(x[1], self.Ly)
        
        vertices = mesh.locate_entities_boundary(
            self.mesh, 0,
            inflow_boundary)
        dofs = fem.locate_dofs_topological(self.P1, 0, vertices)
        
        # 创建入口饱和度边界条件函数
        Sw_bc_func = fem.Function(self.P1)
        Sw_bc_func.x.array[:] = 1.0
        bc = fem.dirichletbc(Sw_bc_func, dofs)
        
        # 组装矩阵和向量
        A = petsc.create_matrix(fem.form(ufl.lhs(F)))
        b = petsc.create_vector(fem.form(ufl.rhs(F)))
        
        petsc.assemble_matrix(A, fem.form(ufl.lhs(F)), bcs=[bc])
        A.assemble()
        petsc.assemble_vector(b, fem.form(ufl.rhs(F)))
        petsc.apply_lifting(b, [fem.form(ufl.lhs(F))], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # 设置求解器
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType("gmres")
        pc = ksp.getPC()
        pc.setType("ilu")
        ksp.setTolerances(rtol=1e-8, max_it=1000)
        
        # 求解系统
        x = A.createVecRight()
        ksp.solve(b, x)
        
        # 将解复制到函数中并同步
        Sw_solution.x.array[:] = x.array[:]
        Sw_solution.x.scatter_forward()
        
        # 确保饱和度在物理范围内
        self.Sw.x.array[:] = np.clip(Sw_solution.x.array, 0.0, 1.0)
        self.Sw.x.scatter_forward()
    
    def setup_wells(self, f):
        """设置井的源汇项"""
        inj_rate = 1e-12  # 注入速率
        prod_rate = -1e-12  # 生产速率
        
        # 定义井的位置（使用几何标记）
        def injection_well(x):
            return np.logical_and(
                np.isclose(x[0], self.Lx, atol=self.Lx/30),
                np.isclose(x[1], self.Ly, atol=self.Ly/30)
            )
            
        def production_well(x):
            return np.logical_and(
                np.isclose(x[0], self.Lx/2, atol=self.Lx/30),
                np.isclose(x[1], self.Ly/2, atol=self.Ly/30)
            )
        
        # 设置源汇项
        f.interpolate(lambda x: np.zeros_like(x[0]))
        
        # 标记井的位置
        inj_dofs = fem.locate_dofs_geometrical(self.P1, injection_well)
        prod_dofs = fem.locate_dofs_geometrical(self.P1, production_well)
        
        # 设置源汇项值
        f.x.array[inj_dofs] = inj_rate
        f.x.array[prod_dofs] = prod_rate
    
    def save_results(self, step, pressure):
        """保存结果"""
        # 确保results目录存在
        if self.comm.rank == 0:
            if not os.path.exists('results'):
                os.makedirs('results')
        # 等待所有进程
        self.comm.barrier()
        
        # 保存为VTK格式
        pressure_file = io.VTXWriter(self.comm, 
                                   f"results/pressure_{step:04d}.bp",
                                   [pressure])
        pressure_file.write(0.0)
        
        sw_file = io.VTXWriter(self.comm,
                              f"results/saturation_{step:04d}.bp",
                              [self.Sw])
        sw_file.write(0.0)
        
        # 保存历史记录
        if self.comm.rank == 0:
            self.pressure_history.append(pressure.x.array.copy())
            self.saturation_history.append(self.Sw.x.array.copy())
            self.time_steps.append(step * self.dt)
    
    def visualize_results(self, pressure, step):
        """可视化结果（仅在主进程上执行）"""
        if self.comm.rank == 0:
            # 创建图形
            plt.figure(figsize=(15, 5))
            
            # 获取中间切片的索引
            z_coords = self.mesh.geometry.x[:, 2]
            z_mid = (z_coords.max() + z_coords.min()) / 2
            mask = np.isclose(z_coords, z_mid, atol=self.Lz/6)  # 增加容差范围
            
            # 检查是否有数据点
            if not np.any(mask):
                print(f"警告：在步骤 {step} 没有找到满足切片条件的数据点")
                return
            
            # 提取中间切片的坐标和数据
            x_slice = self.mesh.geometry.x[mask, 0]
            y_slice = self.mesh.geometry.x[mask, 1]
            p_array = pressure.x.array[mask]
            s_array = self.Sw.x.array[mask]
            
            # 安全地处理非有限值
            if np.any(np.isfinite(p_array)):
                p_max = np.nanmax(p_array[np.isfinite(p_array)])
                p_min = np.nanmin(p_array[np.isfinite(p_array)])
            else:
                p_max = 1.0
                p_min = 0.0
            
            p_array = np.nan_to_num(p_array, nan=0.0, posinf=p_max, neginf=p_min)
            s_array = np.clip(np.nan_to_num(s_array, nan=0.0), 0.0, 1.0)
            
            # 压力分布
            plt.subplot(121)
            scatter = plt.scatter(x_slice, y_slice, c=p_array, cmap='viridis')
            plt.colorbar(scatter, label='Pressure (Pa)')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'Pressure at step {step}')
            
            # 饱和度分布
            plt.subplot(122)
            scatter = plt.scatter(x_slice, y_slice, c=s_array, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(scatter, label='Water Saturation')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'Water Saturation at step {step}')
            
            plt.tight_layout()
            
            # 确保results目录存在
            os.makedirs('results', exist_ok=True)
            
            # 保存图像
            plt.savefig(f'results/visualization_step_{step:04d}.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    # 创建模拟器实例
    simulator = MultiphaseFlowSimulator(
        nx=30, ny=30, nz=3,
        Lx=100.0, Ly=100.0, Lz=20.0,
        k=200e-13,
        phi=0.25,
        Sw_init=0.8,
        dt=43200
    )
    
    # 时间步进
    n_steps = 200
    
    if simulator.comm.rank == 0:
        print("开始模拟...")
    
    for step in range(n_steps):
        if simulator.comm.rank == 0:
            print(f"\r进度: {step+1}/{n_steps}", end="")
        
        # 求解压力方程
        velocity, pressure = simulator.solve_pressure()
        
        # 求解输运方程
        simulator.solve_transport(velocity)
        
        # 每20步保存和可视化结果
        if step % 20 == 0:
            simulator.save_results(step, pressure)
            simulator.visualize_results(pressure, step)
            
            if simulator.comm.rank == 0:
                print(f"\n完成时间步 {step}")
                print(f"  压力范围: [{pressure.x.array.min():.2e}, {pressure.x.array.max():.2e}] Pa")
                print(f"  平均饱和度: {simulator.Sw.x.array.mean():.3f}")
    
    if simulator.comm.rank == 0:
        print("\n模拟完成！")

if __name__ == "__main__":
    main()
