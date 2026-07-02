// FormationManager.cs
// 4 協調取り囲み制御。python/tello_rl/formation.py をミラー。
// 進行方向角 phi のローパス(式40-42)、正多角形(式43-44)/進行方向依存(式45-52)スロット、辺長(式88)。
using System;

namespace TelloEnclosing
{
    public class FormationManager
    {
        private readonly EnvConfig cfg;
        public double phi;
        public V2[] slots;
        public V2 rCgRef;
        public double[] edgeRef;

        public FormationManager(EnvConfig cfg) { this.cfg = cfg; phi = 0.0; }

        public static double WrapPi(double a) => Math.Atan2(Math.Sin(a), Math.Cos(a));

        // 進行方向角更新 (式40-42)
        public void UpdateHeading(V2 vo)
        {
            double speed = vo.Norm();
            if (speed < cfg.v_eps) return;
            double phiRaw = Math.Atan2(vo.y, vo.x);          // 式40
            double dPhi = WrapPi(phiRaw - phi);              // 式41
            phi = phi + cfg.lambda_phi * dPhi;              // 式42
        }

        // Case に応じてスロット・重心参照・辺長を計算
        public void ComputeSlots(V2 ro, V2 vo)
        {
            int N = cfg.N;
            slots = new V2[N];
            if (cfg.caseId == 1)
            {
                for (int i = 0; i < N; i++)
                {
                    double a = phi + cfg.alpha0 + 2.0 * Math.PI * i / N;  // 式43
                    slots[i] = ro + cfg.R * new V2(Math.Cos(a), Math.Sin(a)); // 式44
                }
                rCgRef = ro;                                  // 式87 Case1
            }
            else
            {
                double nu = vo.Norm() / cfg.v_scale;          // 式45
                double alphaRaw = (2.0 * Math.PI / N) * Math.Pow(cfg.cv, -nu);
                double alpha = Math.Max(cfg.alpha_min, alphaRaw); // 式46
                V2 mean = new V2(0, 0);
                for (int i = 0; i < N; i++)
                {
                    double a = phi - (N - 1) / 2.0 * alpha + i * alpha; // 式49
                    slots[i] = ro + cfg.R * new V2(Math.Cos(a), Math.Sin(a)); // 式50
                    mean = mean + (1.0 / N) * slots[i];
                }
                rCgRef = mean;                                // 式52,87 Case2
            }
            // 辺長 (式88): i と i+ の目標辺長
            edgeRef = new double[N];
            for (int i = 0; i < N; i++)
                edgeRef[i] = (slots[i] - slots[(i + 1) % N]).Norm();
        }
    }
}
