#!/usr/bin/env python3
"""Well-conditioned CONTROL denominator census: turbulent_radiative_layer_2D.
Turbulent mixing layer -> all fields should have HEALTHY spatial variance
(none near the epsilon floor) -> the diagnostic correctly reports NO artifact.
Same data-only method + window (frames 4-33) as the RT/RB census."""
import fsspec, h5py, numpy as np, json, time, sys
HF="https://huggingface.co/datasets/polymathic-ai/turbulent_radiative_layer_2D/resolve/main/data/test/"
FILES=["turbulent_radiative_layer_tcool_0.03.hdf5","turbulent_radiative_layer_tcool_0.10.hdf5","turbulent_radiative_layer_tcool_1.00.hdf5"]
WIN0,WIN1=4,33; EPS_FIX=1e-5
def sv(f): 
    f=f.astype(np.float64).reshape(f.shape[0],-1); return f.var(axis=1,ddof=1)
def main():
    fs=fsspec.filesystem("http"); out={"window":[WIN0,WIN1],"files":{}}
    for fn in FILES:
        t=time.time()
        with fs.open(HF+fn,"rb",block_size=4*1024*1024) as fo:
            with h5py.File(fo,"r") as h5:
                den=h5["t0_fields/density"][0,WIN0:WIN1+1]
                pre=h5["t0_fields/pressure"][0,WIN0:WIN1+1]
                vel=h5["t1_fields/velocity"][0,WIN0:WIN1+1]
                dv,pv,vx,vy=sv(den),sv(pre),sv(vel[...,0]),sv(vel[...,1])
                fb=lambda a: float(np.mean(a<=EPS_FIX))
                rec={"density_var":[float(dv.min()),float(dv.max())],"pressure_var":[float(pv.min()),float(pv.max())],
                     "vx_var":[float(vx.min()),float(vx.max())],"vy_var":[float(vy.min()),float(vy.max())],
                     "any_field_below_epsfix":{"density":fb(dv),"pressure":fb(pv),"vx":fb(vx),"vy":fb(vy)},"sec":round(time.time()-t,1)}
                out["files"][fn]=rec
                print(f"{fn}: {rec['sec']}s  density_var[{dv.min():.2e},{dv.max():.2e}] pressure_var[{pv.min():.2e},{pv.max():.2e}] "
                      f"vx_var[{vx.min():.2e},{vx.max():.2e}] vy_var[{vy.min():.2e},{vy.max():.2e}] | "
                      f"frac<=epsfix: den {fb(dv):.0%} pre {fb(pv):.0%} vx {fb(vx):.0%} vy {fb(vy):.0%}")
                sys.stdout.flush()
    json.dump(out,open(".gate-work/control_census_out.json","w"),indent=2)
    print("saved .gate-work/control_census_out.json")
main()
