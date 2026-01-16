LoadPackage("QDistRnd");

hxinfo := ReadMTXE("/Users/leverrie/research/qtanner-search/data/lrz_paper_mtx/633x633/HX_C11_396_2_29.mtx");
hzinfo := ReadMTXE("/Users/leverrie/research/qtanner-search/data/lrz_paper_mtx/633x633/HZ_C11_396_2_29.mtx");
HXm := hxinfo[3];
HZm := hzinfo[3];

Print("Code: C11_396_2_29\n");

Print("DistRandCSS(HX,HZ,num=100000,mindist=0,debug=0)\n");
t0 := Runtime();
dz := DistRandCSS(HXm, HZm, 100000, 0, 0 : field := GF(2));
t1 := Runtime();
Print("dZ = ", dz, "   time_ms=", (t1-t0), "\n");

Print("DistRandCSS(HZ,HX,num=100000,mindist=0,debug=0)\n");
t0 := Runtime();
dx := DistRandCSS(HZm, HXm, 100000, 0, 0 : field := GF(2));
t1 := Runtime();
Print("dX = ", dx, "   time_ms=", (t1-t0), "\n");

QUIT;
