// ObservationBuilder.cs
// 5.1 局所観測 o_i (式82)。python/tello_rl/observation.py をミラー。
// 固定長・距離順近傍・mask・4方向壁距離。obs_dim は config.py と一致 (N=3 で 34)。
using System;
using System.Collections.Generic;

namespace TelloEnclosing
{
    public static class ObservationBuilder
    {
        public static int ObsDim(EnvConfig cfg)
        {
            int nObs = cfg.ResolvedNMaxObs();
            return 2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 5 * nObs + 4 + 1 + 4 + 2;
        }

        public static double[] Build(
            int i, TelloAgent[] agents, V2 ro, V2 vo, V2[] slots,
            double[] aPrev, double battery, EnvConfig cfg)
        {
            int N = cfg.N;
            int nObs = cfg.ResolvedNMaxObs();
            var st = agents[i];
            var f = new List<double>();
            // ri-ro, ri-slot, vi, vo
            Add(f, (st.r - ro), 1.0 / cfg.R_max);
            Add(f, (st.r - slots[i]), 1.0 / cfg.R_max);
            Add(f, st.v, 1.0 / cfg.v_max_norm);
            Add(f, vo, 1.0 / cfg.vo_max_norm);
            // yaw
            f.Add(Math.Sin(st.psi)); f.Add(Math.Cos(st.psi));
            f.Add(st.psidot / cfg.omega_max_psi);
            f.Add((st.h - cfg.h_ref) / cfg.h_scale);
            f.Add(st.hdot / cfg.vz_max_norm);
            // 近傍 (距離順, mask)
            var others = new List<int>();
            for (int j = 0; j < N; j++) if (j != i) others.Add(j);
            others.Sort((p, q) => (agents[p].r - st.r).Norm().CompareTo((agents[q].r - st.r).Norm()));
            for (int ell = 0; ell < nObs; ell++)
            {
                if (ell < others.Count)
                {
                    int j = others[ell];
                    Add(f, (agents[j].r - st.r), 1.0 / cfg.R_max);
                    Add(f, (agents[j].v - st.v), 1.0 / cfg.v_max_norm);
                    f.Add(1.0); // mask
                }
                else { f.Add(0); f.Add(0); f.Add(0); f.Add(0); f.Add(0); }
            }
            // 壁
            double[] wd = SafetyEvaluator.WallDistanceVec(st.r, cfg);
            for (int q = 0; q < 4; q++) f.Add(wd[q] / cfg.d_wall_max);
            // バッテリ
            f.Add(battery / 100.0);
            // 前回入力
            for (int q = 0; q < 4; q++) f.Add(aPrev[q]);
            // 役割
            f.Add(Math.Sin(2 * Math.PI * i / N)); f.Add(Math.Cos(2 * Math.PI * i / N));
            return f.ToArray();
        }

        private static void Add(List<double> f, V2 v, double scale)
        {
            f.Add(v.x * scale); f.Add(v.y * scale);
        }
    }
}
