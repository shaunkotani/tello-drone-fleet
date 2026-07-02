// TelloAgent.cs
// 3.2 SDK コマンドレベルモデル (式(16)-(26))。python/tello_rl/command_model.py をミラー。
// 状態(水平位置・速度・高度・上下速度・yaw・yaw角速度)を保持し1ステップ前進する。
// 可視化したい場合は MonoBehaviour 化し、State を Unity 座標へ写像して描画する。
using System;

namespace TelloEnclosing
{
    public class TelloAgent
    {
        public V2 r, v;          // 水平位置[m], 水平速度(世界)[m/s]
        public double h, hdot;   // 高度[m], 上下速度[m/s]
        public double psi, psidot; // yaw[rad], yaw角速度[rad/s]

        public TelloAgent(V2 r0, double psi0, double h0)
        {
            r = r0; v = new V2(0, 0); h = h0; hdot = 0; psi = psi0; psidot = 0;
        }

        // 正規化行動の前後・左右成分 -> 世界座標の水平速度指令 (式(18),(20))
        // a = [a_lr, a_fb, a_ud, a_yaw]
        public static V2 ActionToWorldVelCmd(double[] a, double psi, EnvConfig cfg)
        {
            double vbx = cfg.v_max_fb * a[1];   // 前後
            double vby = cfg.v_max_lr * a[0];   // 左右
            double c = Math.Cos(psi), s = Math.Sin(psi);
            return new V2(c * vbx - s * vby, s * vbx + c * vby); // 式(20)
        }

        // 1 ステップ前進 (式(21)-(26))。a は executed action in [-1,1]^4。
        public void Step(double[] a, EnvConfig cfg, System.Random rng)
        {
            double dt = cfg.dt;
            V2 vcmd = ActionToWorldVelCmd(a, psi, cfg);
            double vzcmd = cfg.v_max_ud * a[2];          // 式(19)
            double psidotcmd = cfg.omega_max_psi * a[3]; // 式(19)

            V2 wv = new V2(0, 0);
            if (rng != null && cfg.wind_acc_std > 0)
                wv = new V2(Gauss(rng) * cfg.wind_acc_std, Gauss(rng) * cfg.wind_acc_std);

            // 水平 (式21,22)
            v = v + (dt / cfg.tau_v) * (vcmd - v) + dt * wv;
            r = r + dt * v;
            // 高度 (式23,24)
            hdot = hdot + (dt / cfg.tau_z) * (vzcmd - hdot);
            h = h + dt * hdot;
            // yaw (式25,26)
            psidot = psidot + (dt / cfg.tau_psi) * (psidotcmd - psidot);
            psi = psi + dt * psidot;
        }

        private static double Gauss(System.Random rng)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
