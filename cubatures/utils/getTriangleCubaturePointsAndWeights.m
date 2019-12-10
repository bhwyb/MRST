function [x, w, n, xR] = getTriangleCubaturePointsAndWeights(k)
    % Get triangle cubature points and weights for a cubature of precision k,
    % taken from the book "P. Solin, K. Segeth and I. Dolezel: Higher-Order
    % Finite Element Methods", Chapman & Hall/CRC Press, 2003.
    %
    % SYNOPSIS:
    %
    %   [x, w, n, xR] = getTriangleCubaturePointsAndWeights(k)
    %
    % PARAMETERS:
    %   k - cubature prescision
    %
    % RETURNS:
    %   x  - Cubature points
    %   w  - Cubature weights
    %   n  - Number of cubature points
    %   xR - Coordinates of reference triangle
    
    xR = [-1, -1;
           1, -1; 
          -1,  1];

    if k <= 1

        xw = [  -0.333333333333333  -0.333333333333333  2.000000000000000];

    elseif k <= 2

        xw = [  -0.666666666666667  -0.666666666666667  0.666666666666667
                -0.666666666666667   0.333333333333333  0.666666666666667
                 0.333333333333333  -0.666666666666667  0.666666666666667];

    elseif k <= 3

        xw = [  -0.333333333333333  -0.333333333333333  -1.125000000000000
                -0.600000000000000  -0.600000000000000   1.041666666666667
                -0.600000000000000   0.200000000000000   1.041666666666667
                 0.200000000000000  -0.600000000000000   1.041666666666667];

    elseif k <= 4

        xw = [  -0.108103018168070  -0.108103018168070   0.446763179356022
                -0.108103018168070  -0.783793963663860   0.446763179356022
                -0.783793963663860  -0.108103018168070   0.446763179356022
                -0.816847572980458  -0.816847572980458   0.219903487310644
                -0.816847572980458   0.633695145960918   0.219903487310644
                 0.633695145960918  -0.816847572980458   0.219903487310644];

    elseif k <= 5

        xw = [  -0.333333333333333  -0.333333333333333  0.450000000000000
                -0.059715871789770  -0.059715871789770  0.264788305577012
                -0.059715871789770  -0.880568256420460  0.264788305577012
                -0.880568256420460  -0.059715871789770  0.264788305577012
                -0.797426985353088  -0.797426985353088  0.251878361089654
                -0.797426985353088  0.594853970706174  0.251878361089654
                0.594853970706174  -0.797426985353088  0.251878361089654];

    elseif k <= 6

        xw = [  -0.501426509658180  -0.501426509658180  0.233572551452758
                -0.501426509658180  0.002853019316358  0.233572551452758
                0.002853019316358  -0.501426509658180  0.233572551452758
                -0.873821971016996  -0.873821971016996  0.101689812740414
                -0.873821971016996  0.747643942033992  0.101689812740414
                0.747643942033992  -0.873821971016996  0.101689812740414
                -0.379295097932432  0.273004998242798  0.165702151236748
                0.273004998242798  -0.893709900310366  0.165702151236748
                -0.893709900310366  -0.379295097932432  0.165702151236748
                -0.379295097932432  -0.893709900310366  0.165702151236748
                0.273004998242798  -0.379295097932432  0.165702151236748
                -0.893709900310366  0.273004998242798  0.165702151236748];

    elseif k <= 7

        xw = [  -0.333333333333333  -0.333333333333333  -0.299140088935364
                -0.479308067841920  -0.479308067841920  0.351230514866416
                -0.479308067841920  -0.041383864316160  0.351230514866416
                -0.041383864316160  -0.479308067841920  0.351230514866416
                -0.869739794195568  -0.869739794195568  0.106694471217676
                -0.869739794195568  0.739479588391136  0.106694471217676
                0.739479588391136  -0.869739794195568  0.106694471217676
                -0.374269007990252  0.276888377139620  0.154227521780514
                0.276888377139620  -0.902619369149368  0.154227521780514
                -0.902619369149368  -0.374269007990252  0.154227521780514
                -0.374269007990252  -0.902619369149368  0.154227521780514
                0.276888377139620  -0.374269007990252  0.154227521780514
                -0.902619369149368  0.276888377139620  0.154227521780514];


    elseif k <= 8

       xw = [   -0.333333333333333  -0.333333333333333  0.288631215355574
                -0.081414823414554  -0.081414823414554  0.190183268534570
                -0.081414823414554  -0.837170353170892  0.190183268534570
                -0.837170353170892  -0.081414823414554  0.190183268534570
                -0.658861384496480  -0.658861384496480  0.206434741069436
                -0.658861384496480  0.317722768992960  0.206434741069436
                0.317722768992960  -0.658861384496480  0.206434741069436
                -0.898905543365938  -0.898905543365938  0.064916995246396
                -0.898905543365938  0.797811086731876  0.064916995246395
                0.797811086731876  -0.898905543365938  0.064916995246396
                -0.473774340730724  0.456984785910808  0.054460628348870
                0.456984785910808  -0.983210445180084  0.054460628348870
                -0.983210445180084  -0.473774340730724  0.054460628348870
                -0.473774340730724  -0.983210445180084  0.054460628348870
                0.456984785910808  -0.473774340730724  0.054460628348870
                -0.983210445180084  0.456984785910808  0.054460628348870];


    elseif k <= 9
        xw = [  -0.333333333333333  -0.333333333333333  0.194271592565598
                -0.020634961602524  -0.020634961602524  0.062669400454278
                -0.020634961602524  -0.958730076794950  0.062669400454278
                -0.958730076794950  -0.020634961602524  0.062669400454278
                -0.125820817014126  -0.125820817014126  0.155655082009548
                -0.125820817014126  -0.748358365971746  0.155655082009548
                -0.748358365971746  -0.125820817014126  0.155655082009548
                -0.623592928761934  -0.623592928761934  0.159295477854420
                -0.623592928761934  0.247185857523870  0.159295477854420
                0.247185857523870  -0.623592928761934  0.159295477854420
                -0.910540973211094  -0.910540973211094  0.051155351317396
                -0.910540973211094  0.821081946422190  0.051155351317396
                0.821081946422190  -0.910540973211094  0.051155351317396
                -0.556074021678468  0.482397197568996  0.086567078754578
                0.482397197568996  -0.926323175890528  0.086567078754578
                -0.926323175890528  -0.556074021678468  0.086567078754578
                -0.556074021678468  -0.926323175890528  0.086567078754578
                0.482397197568996  -0.556074021678468  0.086567078754578
                -0.926323175890528  0.482397197568996  0.086567078754578];

    elseif k <= 10

        xw = [  -0.333333333333333  -0.333333333333333  0.181635980765508
                -0.028844733232686  -0.028844733232686  0.073451915512934
                -0.028844733232686  -0.942310533534630  0.073451915512934
                -0.942310533534630  -0.028844733232686  0.073451915512934
                -0.781036849029926  -0.781036849029926  0.090642118871056
                -0.781036849029926  0.562073698059852  0.090642118871056
                0.562073698059852  -0.781036849029926  0.090642118871056
                -0.384120322471758  0.100705883641998  0.145515833690840
                0.100705883641998  -0.716585561170240  0.145515833690840
                -0.716585561170240  -0.384120322471758  0.145515833690840
                -0.384120322471758  -0.716585561170240  0.145515833690840
                0.100705883641998  -0.384120322471758  0.145515833690840
                -0.716585561170240  0.100705883641998  0.145515833690840
                -0.506654878720194  0.456647809194822  0.056654485062114
                0.456647809194822  -0.949992930474628  0.056654485062114
                -0.949992930474628  -0.506654878720194  0.056654485062114
                -0.506654878720194  -0.949992930474628  0.056654485062114
                0.456647809194822  -0.506654878720194  0.056654485062114
                -0.949992930474628  0.456647809194822  0.056654485062114
                -0.866393497975600  0.847311867175000  0.018843333927466
                0.847311867175000  -0.980918369199402  0.018843333927466
                -0.980918369199402  -0.866393497975600  0.018843333927466
                -0.866393497975600  -0.980918369199402  0.018843333927466
                0.847311867175000  -0.866393497975600  0.018843333927466
                -0.980918369199402  0.847311867175000  0.018843333927466];

    elseif k <= 11

        xw = [  0.069222096541516  0.069222096541516  0.001854012657922
                0.069222096541516  -1.138444193083034  0.001854012657922
                -1.138444193083034  0.069222096541516  0.001854012657922
                -0.202061394068290  -0.202061394068290  0.154299069829626
                -0.202061394068290  -0.595877211863420  0.154299069829626
                -0.595877211863420  -0.202061394068290  0.154299069829626
                -0.593380199137436  -0.593380199137436  0.118645954761548
                -0.593380199137436  0.186760398274870  0.118645954761548
                0.186760398274870  -0.593380199137436  0.118645954761548
                -0.761298175434838  -0.761298175434838  0.072369081006836
                -0.761298175434838  0.522596350869674  0.072369081006836
                0.522596350869674  -0.761298175434838  0.072369081006836
                -0.935270103777448  -0.935270103777448  0.027319462005356
                -0.935270103777448  0.870540207554896  0.027319462005356
                0.870540207554896  -0.935270103777448  0.027319462005356
                -0.286758703477414  0.186402426856426  0.104674223924408
                0.186402426856426  -0.899643723379010  0.104674223924408
                -0.899643723379010  -0.286758703477414  0.104674223924408
                -0.286758703477414  -0.899643723379010  0.104674223924408
                0.186402426856426  -0.286758703477414  0.104674223924408
                -0.899643723379010  0.186402426856426  0.104674223924408
                -0.657022039391916  0.614978006319584  0.041415319278282
                0.614978006319584  -0.957955966927668  0.041415319278282
                -0.957955966927668  -0.657022039391916  0.041415319278282
                -0.657022039391916  -0.957955966927668  0.041415319278282
                0.614978006319584  -0.657022039391916  0.041415319278282
                -0.957955966927668  0.614978006319584  0.041415319278282];

    elseif k <= 12

        xw = [  -0.023565220452390  -0.023565220452390  0.051462132880910
                -0.023565220452390  -0.952869559095220  0.051462132880910
                -0.952869559095220  -0.023565220452390  0.051462132880910
                -0.120551215411080  -0.120551215411080  0.087385089076076
                -0.120551215411080  -0.758897569177842  0.087385089076076
                -0.758897569177842  -0.120551215411080  0.087385089076076
                -0.457579229975768  -0.457579229975768  0.125716448435770
                -0.457579229975768  -0.084841540048464  0.125716448435770
                -0.084841540048464  -0.457579229975768  0.125716448435770
                -0.744847708916828  -0.744847708916828  0.069592225861418
                -0.744847708916828  0.489695417833656  0.069592225861418
                0.489695417833656  -0.744847708916828  0.069592225861418
                -0.957365299093580  -0.957365299093580  0.012332522103118
                -0.957365299093580  0.914730598187158  0.012332522103118
                0.914730598187158  -0.957365299093580  0.012332522103118
                -0.448573460628972  0.217886471559576  0.080743115532762
                0.217886471559576  -0.769313010930604  0.080743115532762
                -0.769313010930604  -0.448573460628972  0.080743115532762
                -0.448573460628972  -0.769313010930604  0.080743115532762
                0.217886471559576  -0.448573460628972  0.080743115532762
                -0.769313010930604  0.217886471559576  0.080743115532762
                -0.437348838020120  0.391672173575606  0.044713546404606
                0.391672173575606  -0.954323335555486  0.044713546404606
                -0.954323335555486  -0.437348838020120  0.044713546404606
                -0.437348838020120  -0.954323335555486  0.044713546404606
                0.391672173575606  -0.437348838020120  0.044713546404606
                -0.954323335555486  0.391672173575606  0.044713546404606
                -0.767496168184806  0.716028067088146  0.034632462217318
                0.716028067088146  -0.948531898903340  0.034632462217318
                -0.948531898903340  -0.767496168184806  0.034632462217318
                -0.767496168184806  -0.948531898903340  0.034632462217318
                0.716028067088146  -0.767496168184806  0.034632462217318
                -0.948531898903340  0.716028067088146  0.034632462217318];

    elseif k <= 13

        xw = [  -0.333333333333333  -0.333333333333333  0.105041846801604
                -0.009903630120590  -0.009903630120590  0.022560290418660
                -0.009903630120590  -0.980192739758818  0.022560290418660
                -0.980192739758818  -0.009903630120590  0.022560290418660
                -0.062566729780852  -0.062566729780852  0.062847036724908
                -0.062566729780852  -0.874866540438296  0.062847036724908
                -0.874866540438296  -0.062566729780852  0.062847036724908
                -0.170957326397446  -0.170957326397446  0.094145005008388
                -0.170957326397446  -0.658085347205106  0.094145005008388
                -0.658085347205106  -0.170957326397446  0.094145005008388
                -0.541200855914338  -0.541200855914338  0.094727173072710
                -0.541200855914338  0.082401711828674  0.094727173072710
                0.082401711828674  -0.541200855914338  0.094727173072710
                -0.771151009607340  -0.771151009607340  0.062335058091588
                -0.771151009607340  0.542302019214680  0.062335058091588
                0.542302019214680  -0.771151009607340  0.062335058091588
                -0.950377217273082  -0.950377217273082  0.015951542930148
                -0.950377217273082  0.900754434546164  0.015951542930148
                0.900754434546164  -0.950377217273082  0.015951542930148
                -0.462410005882478  0.272702349123320  0.073696805457464
                0.272702349123320  -0.810292343240842  0.073696805457464
                -0.810292343240842  -0.462410005882478  0.073696805457464
                -0.462410005882478  -0.810292343240842  0.073696805457464
                0.272702349123320  -0.462410005882478  0.073696805457464
                -0.810292343240842  0.272702349123320  0.073696805457464
                -0.416539866531424  0.380338319973810  0.034802926607644
                0.380338319973810  -0.963798453442386  0.034802926607644
                -0.963798453442386  -0.416539866531424  0.034802926607644
                -0.416539866531424  -0.963798453442386  0.034802926607644
                0.380338319973810  -0.416539866531424  0.034802926607644
                -0.963798453442386  0.380338319973810  0.034802926607644
                -0.747285229016662  0.702819075668482  0.031043573678090
                0.702819075668482  -0.955533846651820  0.031043573678090
                -0.955533846651820  -0.747285229016662  0.031043573678090
                -0.747285229016662  -0.955533846651820  0.031043573678090
                0.702819075668482  -0.747285229016662  0.031043573678090
                -0.955533846651820  0.702819075668482  0.031043573678090];

    elseif k <= 14

        xw = [  -0.022072179275642  -0.022072179275642  0.043767162738858
                -0.022072179275642  -0.955855641448714  0.043767162738858
                -0.955855641448714  -0.022072179275642  0.043767162738858
                -0.164710561319092  -0.164710561319092  0.065576707088250
                -0.164710561319092  -0.670578877361816  0.065576707088250
                -0.670578877361816  -0.164710561319092  0.065576707088250
                -0.453044943382322  -0.453044943382322  0.103548209014584
                -0.453044943382322  -0.093910113235354  0.103548209014584
                -0.093910113235354  -0.453044943382322  0.103548209014584
                -0.645588935174914  -0.645588935174914  0.084325177473986
                -0.645588935174914  0.291177870349826  0.084325177473986
                0.291177870349826  -0.645588935174914  0.084325177473986
                -0.876400233818254  -0.876400233818254  0.028867399339554
                -0.876400233818254  0.752800467636510  0.028867399339554
                0.752800467636510  -0.876400233818254  0.028867399339554
                -0.961218077502598  -0.961218077502598  0.009846807204800
                -0.961218077502598  0.922436155005196  0.009846807204800
                0.9224361550051960  -0.961218077502598  0.009846807204800
                -0.655466624357288  0.541217109549992  0.049331506425128
                0.541217109549992  -0.885750485192704  0.049331506425128
                -0.885750485192704  -0.655466624357288  0.049331506425128
                -0.655466624357288  -0.885750485192704  0.049331506425128
                0.541217109549992  -0.655466624357288  0.049331506425128
                -0.885750485192704  0.541217109549992  0.049331506425128
                -0.326277080407310  0.140444581693366  0.077143021574122
                0.140444581693366  -0.814167501286056  0.077143021574122
                -0.814167501286056  -0.326277080407310  0.077143021574122
                -0.326277080407310  -0.814167501286056  0.077143021574122
                0.140444581693366  -0.326277080407310  0.077143021574122
                -0.814167501286056  0.140444581693366  0.077143021574122
                -0.403254235727484  0.373960335616176  0.028872616227068
                0.373960335616176  -0.970706099888692  0.028872616227068
                -0.970706099888692  -0.403254235727484  0.028872616227068
                -0.403254235727484  -0.970706099888692  0.028872616227068
                0.373960335616176  -0.403254235727484  0.028872616227068
                -0.970706099888692  0.373960335616176  0.028872616227068
                -0.762051004606086  0.759514342740342  0.010020457677002
                0.759514342740342  -0.997463338134256  0.010020457677002
                -0.997463338134256  -0.762051004606086  0.010020457677002
                -0.762051004606086  -0.997463338134256  0.010020457677002
                0.759514342740342  -0.762051004606086  0.010020457677002
                -0.997463338134256  0.759514342740342  0.010020457677002];

    elseif k <= 15

        xw = [  0.013945833716486  0.013945833716486  0.003833751285698
                0.013945833716486  -1.027891667432972  0.003833751285698
                -1.027891667432972  0.013945833716486  0.003833751285698
                -0.137187291433954  -0.137187291433954  0.088498054542290
                -0.137187291433954  -0.725625417132090  0.088498054542290
                -0.725625417132090  -0.137187291433954  0.088498054542290
                -0.444612710305712  -0.444612710305712  0.102373097437704
                -0.444612710305712  -0.110774579388578  0.102373097437704
                -0.110774579388578  -0.444612710305712  0.102373097437704
                -0.747070217917492  -0.747070217917492  0.047375471741376
                -0.747070217917492  0.494140435834984  0.047375471741376
                0.494140435834984  -0.747070217917492  0.047375471741376
                -0.858383228050628  -0.858383228050628  0.026579551380042
                -0.858383228050628  0.716766456101256  0.026579551380042
                0.716766456101256  -0.858383228050628  0.026579551380042
                -0.962069659517854  -0.962069659517854  0.009497833216384
                -0.962069659517854  0.924139319035706  0.009497833216384
                0.924139319035706  -0.962069659517854  0.009497833216384
                -0.477377257719826  0.209908933786582  0.077100145199186
                0.209908933786582  -0.732531676066758  0.077100145199186
                -0.732531676066758  -0.477377257719826  0.077100145199186
                -0.477377257719826  -0.732531676066758  0.077100145199186
                0.209908933786582  -0.477377257719826  0.077100145199186
                -0.732531676066758  0.209908933786582  0.077100145199186
                -0.223906465819462  0.151173111025628  0.054431628641248
                0.151173111025628  -0.927266645206166  0.054431628641248
                -0.927266645206166  -0.223906465819462  0.054431628641248
                -0.223906465819462  -0.927266645206166  0.054431628641248
                0.151173111025628  -0.223906465819462  0.054431628641248
                -0.927266645206166  0.151173111025628  0.054431628641248
                -0.428575559900168  0.448925326153310  0.004364154733594
                0.448925326153310  -1.020349766253142  0.004364154733594
                -1.020349766253142  -0.428575559900168  0.004364154733594
                -0.428575559900168  -1.020349766253142  0.004364154733594
                0.448925326153310  -0.428575559900168  0.004364154733594
                -1.020349766253142  0.448925326153310  0.004364154733594
                -0.568800671855432  0.495112932103676  0.043010639695462
                0.495112932103676  -0.926312260248244  0.043010639695462
                -0.926312260248244  -0.568800671855432  0.043010639695462
                -0.568800671855432  -0.926312260248244  0.043010639695462
                0.495112932103676  -0.568800671855432  0.043010639695462
                -0.926312260248244  0.495112932103676  0.043010639695462
                -0.792848766847228  0.767929148184832  0.015347885262098
                0.767929148184832  -0.975080381337602  0.015347885262098
                -0.975080381337602  -0.792848766847228  0.015347885262098
                -0.792848766847228  -0.975080381337602  0.015347885262098
                0.767929148184832  -0.792848766847228  0.015347885262098
                -0.975080381337602  0.767929148184832  0.015347885262098];

    elseif k <= 16

        xw = [  -0.333333333333333  -0.333333333333333  0.093751394855284
                -0.005238916103124  -0.005238916103124  0.012811757157170
                -0.005238916103124  -0.989522167793754  0.012811757157170
                -0.989522167793754  -0.005238916103124  0.012811757157170
                -0.173061122901296  -0.173061122901296  0.083420593478774
                -0.173061122901296  -0.653877754197410  0.083420593478774
                -0.653877754197410  -0.173061122901296  0.083420593478774
                -0.059082801866018  -0.059082801866018  0.053782968500128
                -0.059082801866018  -0.881834396267966  0.053782968500128
                -0.881834396267966  -0.059082801866018  0.053782968500128
                -0.518892500060958  -0.518892500060958  0.084265045523300
                -0.518892500060958  0.037785000121916  0.084265045523300
                0.037785000121916  -0.518892500060958  0.084265045523300
                -0.704068411554854  -0.704068411554854  0.060000533685546
                -0.704068411554854  0.408136823109708  0.060000533685546
                0.408136823109708  -0.704068411554854  0.060000533685546
                -0.849069624685052  -0.849069624685052  0.028400197850048
                -0.849069624685052  0.698139249370104  0.028400197850048
                0.698139249370104  -0.849069624685052  0.028400197850048
                -0.966807194753950  -0.966807194753950  0.007164924702546
                -0.966807194753950  0.933614389507900  0.007164924702546
                0.933614389507900  -0.966807194753950  0.007164924702546
                -0.406888806840226  0.199737422349722  0.065546294921254
                0.199737422349722  -0.792848615509496  0.065546294921254
                -0.792848615509496  -0.406888806840226  0.065546294921254
                -0.406888806840226  -0.792848615509496  0.065546294921254
                0.199737422349722  -0.406888806840226  0.065546294921254
                -0.792848615509496  0.199737422349722  0.065546294921254
                -0.324553873193842  0.284387049883010  0.030596612496882
                0.284387049883010  -0.959833176689168  0.030596612496882
                -0.959833176689168  -0.324553873193842  0.030596612496882
                -0.324553873193842  -0.959833176689168  0.030596612496882
                0.284387049883010  -0.324553873193842  0.030596612496882
                -0.959833176689168  0.284387049883010  0.030596612496882
                -0.590503436714376  0.599185441942654  0.004772488385678
                0.599185441942654  -1.008682005228278  0.004772488385678
                -1.008682005228278  -0.590503436714376  0.004772488385678
                -0.590503436714376  -1.008682005228278  0.004772488385678
                0.599185441942654  -0.590503436714376  0.004772488385678
                -1.008682005228278  0.599185441942654  0.004772488385678
                -0.621283015738754  0.537399442802736  0.038169585511798
                0.537399442802736  -0.916116427063980  0.038169585511798
                -0.916116427063980  -0.621283015738754  0.038169585511798
                -0.621283015738754  -0.916116427063980  0.038169585511798
                0.537399442802736  -0.621283015738754  0.038169585511798
                -0.916116427063980  0.537399442802736  0.038169585511798
                -0.829432768634686  0.800798128173322  0.013700109093084
                0.800798128173322  -0.971365359538638  0.013700109093084
                -0.971365359538638  -0.829432768634686  0.013700109093084
                -0.829432768634686  -0.971365359538638  0.013700109093084
                0.800798128173322  -0.829432768634686  0.013700109093084
                -0.971365359538638  0.800798128173322  0.013700109093084];

    else

        error('Precision not supported');

    end

    x = xw(:, 1:2);
    w = xw(:, 3  )/2;
    n = numel(w);

    end