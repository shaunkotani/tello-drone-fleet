// SafetyEvaluator.cs
// 5.2 目的コスト(式92) と 5.3 安全制約コスト(式95-99)、4.3 参照速度(式53)。
// python/tello_rl/costs.py と nsb.reference_velocity/clip_reference をミラー。
// CTDE: 全状態から各エージェントのコスト・制約を集中評価する。
using System;

namespace TelloEnclosing
{
    public static class SafetyEvaluator
    {
        // 4方向壁余裕 (式81)
        public static double[] WallDistanceVec(V2 r, EnvConfig cfg)
        {
            return new double[] {
                r.x - cfg.x_min, cfg.x_max - r.x,
                r.y - cfg.y_min, cfg.y_max - r.y };
        }

        // 参照速度 v_ref (式53) を有限差分で計算
        public static V2[] ReferenceVelocity(V2[] slotsK, V2[] slotsKp1, V2[] r, EnvConfig cfg)
        {
            int N = cfg.N;
            V2[] vref = new V2[N];
            for (int i = 0; i < N; i++)
            {
                V2 vff = (1.0 / cfg.dt) * (slotsKp1[i] - slotsK[i]);
                vref[i] = vff + cfg.Kp_slot * (slotsK[i] - r[i]); // 式53
            }
            return vref;
        }

        // 参照速度飽和 (式91)
        public static V2 ClipReference(V2 v, EnvConfig cfg)
        {
            double n = v.Norm();
            double scale = Math.Min(1.0, cfg.v_max_norm / Math.Max(n, 1e-3));
            return scale * v;
        }

        // yaw 参照 (式90)
        public static double YawReference(int i, V2[] rNext, V2 roNext, double phi, EnvConfig cfg)
        {
            if (cfg.yaw_ref_mode == "heading") return phi;
            V2 d = roNext - rNext[i];
            return Math.Atan2(d.y, d.x);
        }

        // 目的コスト l_i (式92)
        public static double ObjectiveCost(
            int i, V2[] rNext, V2 roNext, V2[] slotsNext, V2 rCgRefNext, double[] edgeRefNext,
            V2 vxyCmd, V2 vrefClip, double[] a, double[] aPrev, double hNext,
            double psiNext, double psiRefNext, EnvConfig cfg)
        {
            int N = cfg.N;
            V2 ri = rNext[i];
            double R = cfg.R;
            V2 rbar = new V2(0, 0);
            for (int j = 0; j < N; j++) rbar = rbar + (1.0 / N) * rNext[j];

            double cSlot = cfg.w_slot * Sq((ri - slotsNext[i]).Norm()) / (R * R);
            double cR = cfg.w_R * Sq(((ri - roNext).Norm() - R) / R);
            double cC = cfg.w_c * Sq((rbar - rCgRefNext).Norm()) / (R * R);

            double cE = 0.0;
            int im = (i - 1 + N) % N, ip = (i + 1) % N;
            // i+ 側
            double ellp = Math.Max(edgeRefNext[i], 1e-6);
            cE += Sq(((ri - rNext[ip]).Norm() - ellp) / ellp);
            // i- 側
            double ellm = Math.Max(edgeRefNext[im], 1e-6);
            cE += Sq(((ri - rNext[im]).Norm() - ellm) / ellm);
            cE *= cfg.w_e;

            double cV = cfg.w_v * Sq((vxyCmd - vrefClip).Norm()) / (cfg.v_max_norm * cfg.v_max_norm);
            double cU = cfg.w_u * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
            double cDu = 0.0;
            for (int q = 0; q < 4; q++) cDu += (a[q] - aPrev[q]) * (a[q] - aPrev[q]);
            cDu *= cfg.w_du;
            double cH = cfg.w_h * Sq((hNext - cfg.h_ref) / cfg.h_scale);
            double cPsi = cfg.w_psi * (1.0 - Math.Cos(psiNext - psiRefNext));

            return cSlot + cR + cC + cE + cV + cU + cDu + cH + cPsi;
        }

        // 制約コストベクトル C_i in R^5 (式95-99)
        public static double[] ConstraintCost(
            int i, V2[] rNext, V2 roNext, double hNext, double[] a, EnvConfig cfg)
        {
            int N = cfg.N;
            V2 ri = rNext[i];
            double dmin = double.MaxValue;
            for (int j = 0; j < N; j++)
                if (j != i) dmin = Math.Min(dmin, (ri - rNext[j]).Norm());
            double c1 = Math.Max(0.0, 1.0 - dmin / cfg.d_drone);
            double c2 = Math.Max(0.0, 1.0 - (ri - roNext).Norm() / cfg.d_target);
            double[] wd = WallDistanceVec(ri, cfg);
            double dwallMin = Math.Min(Math.Min(wd[0], wd[1]), Math.Min(wd[2], wd[3]));
            double c3 = Math.Max(0.0, 1.0 - dwallMin / cfg.d_safe_wall);
            double c4 = Math.Max(0.0, (Math.Abs(hNext - cfg.h_ref) - cfg.e_h) / cfg.e_h);
            double amax = 0.0; for (int q = 0; q < 4; q++) amax = Math.Max(amax, Math.Abs(a[q]));
            double c5 = Math.Max(0.0, (amax - cfg.a_safe) / (1.0 - cfg.a_safe));
            return new double[] { c1, c2, c3, c4, c5 };
        }

        private static double Sq(double x) => x * x;
    }
}
