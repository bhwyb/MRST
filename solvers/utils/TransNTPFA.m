function T = TransNTPFA(G, u, OSflux)

    dispif(mrstVerbose, 'TransNTPFA... ');
    timer = tic;

    T = cell(2, 1);

    internal = 1:G.faces.num;
    internal(~all(G.faces.neighbors ~= 0, 2)) = [];

    tii = cell(2, 1);
    tsp = cell(2, 1);
    r = cell(2, 1);
    mu = cell(2, 1);

    for j = 1:2
        tend = zeros(G.faces.num, 1);
        tii{j} = zeros(G.faces.num, 2);
        nij = zeros(G.faces.num, 1);

        % Sweep for sparsity set up
        for i = internal
            t = OSflux{i, j};            
            nij(i) = max(0, size(t, 1) - 3);
            tii{j}(i, 1:2) = [t(1, 2), t(2, 2)];
        end

        % Set up sparsity pattern and values
        ii = zeros(sum(nij), 1);
        jj = zeros(sum(nij), 1);
        vv = zeros(sum(nij), 1);
        s = [1; cumsum(nij) + 1];

        % Don't loop over zero rows
        nzrows = 1:G.faces.num;
        nzrows(nij == 0) = [];

        for i = nzrows
            t = OSflux{i, j};
            idx = s(i):(s(i+1) - 1);
            ii(idx) = i;
            jj(idx) = t(3:end-1, 1);
            vv(idx) = t(3:end-1, 2);
            tend(i) = t(end, 2);
        end

        tsp{j} = sparse(ii, jj, vv, G.faces.num, G.cells.num);

        r{j} = tsp{j} * u + tend;
        mu{j} = 0 * r{j} + 0.5;
        T{j} = 0 * r{j};
    end

    epstol = 1e-12 * max(full(max(tsp{1}, [], 2)), full(max(tsp{2}, [], 2)));
    for j = 1:2
        ir = abs(r{j}) <= epstol;
        r{j}(ir) = 0;
    end
    jj = abs(r{1} + r{2}) > epstol;
    mu{1}(jj) = r{2}(jj) ./ (r{1}(jj) + r{2}(jj));
    mu{2}(jj) = ones(sum(jj), 1) - mu{1}(jj);
    assert(all(mu{1} >= 0.0))
    assert(all(mu{2} >= 0.0))
    assert(all(mu{1} <= 1.0))
    assert(all(mu{2} <= 1.0))

    T{1}(internal) = mu{1}(internal) .* tii{1}(internal, 1) + mu{2}(internal) .* tii{2}(internal, 2);
    T{2}(internal) = mu{1}(internal) .* tii{1}(internal, 2) + mu{2}(internal) .* tii{2}(internal, 1);

    dispif(mrstVerbose, 'done in %1.2f s\n', toc(timer));
end
