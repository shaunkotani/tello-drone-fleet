// TargetObject.cs
// 対象物軌道 (8.2-8.4)。python/tello_rl/target.py をミラー。静止/等速/円/折れ線。
using System;

namespace TelloEnclosing
{
    public class TargetObject
    {
        public V2 ro, vo;
        private int k;
        private readonly EnvConfig cfg;

        public TargetObject(EnvConfig cfg) { this.cfg = cfg; Reset(); }

        public void Reset()
        {
            k = 0;
            switch (cfg.target_mode)
            {
                case "circle":
                    ro = new V2(cfg.circle_A, 0.0);
                    vo = new V2(0.0, cfg.circle_A * cfg.circle_omega);
                    break;
                case "const_vel":
                    ro = new V2(-1.0, 0.0); vo = new V2(cfg.v_tar, 0.0); break;
                case "polyline":
                    ro = new V2(0, 0); vo = new V2(cfg.v_tar, 0.0); break;
                default:
                    ro = new V2(0, 0); vo = new V2(0, 0); break;
            }
        }

        public void Step()
        {
            k += 1;
            double dt = cfg.dt;
            switch (cfg.target_mode)
            {
                case "circle": // 式(139)
                    {
                        double A = cfg.circle_A, w = cfg.circle_omega, t = w * k * dt;
                        ro = new V2(A * Math.Cos(t), A * Math.Sin(t));
                        vo = new V2(-A * w * Math.Sin(t), A * w * Math.Cos(t));
                    }
                    break;
                case "polyline": // 式(140)
                    {
                        int seg = (k / cfg.poly_K_seg) % 4;
                        double v = cfg.v_tar;
                        vo = seg == 0 ? new V2(v, 0) : seg == 1 ? new V2(0, v)
                           : seg == 2 ? new V2(-v, 0) : new V2(0, -v);
                        ro = ro + dt * vo;
                    }
                    break;
                case "const_vel": // 式(136)
                    ro = ro + dt * vo;
                    if (ro.x > cfg.x_max - 0.5 || ro.x < cfg.x_min + 0.5)
                        vo = new V2(-vo.x, vo.y);
                    break;
                default: break; // static
            }
        }
    }
}
