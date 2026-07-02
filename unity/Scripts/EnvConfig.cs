// EnvConfig.cs
// 環境設定。python/tello_rl/config.py の EnvConfig をミラー。
// PythonBridge が {"cmd":"config"} を受けるとここへ反映する。
using System;
using Newtonsoft.Json.Linq;

namespace TelloEnclosing
{
    // 倍精度 2D ベクトル (Python の numpy(2,) に対応)
    public struct V2
    {
        public double x, y;
        public V2(double x, double y) { this.x = x; this.y = y; }
        public static V2 operator +(V2 a, V2 b) => new V2(a.x + b.x, a.y + b.y);
        public static V2 operator -(V2 a, V2 b) => new V2(a.x - b.x, a.y - b.y);
        public static V2 operator *(double s, V2 a) => new V2(s * a.x, s * a.y);
        public double Norm() => Math.Sqrt(x * x + y * y);
        public double Dot(V2 b) => x * b.x + y * b.y;
        public double[] ToArray() => new double[] { x, y };
    }

    [Serializable]
    public class EnvConfig
    {
        public int N = 3;
        public double x_min = -2.5, x_max = 2.5, y_min = -2.5, y_max = 2.5;
        public double z_min = 0.0, z_max = 2.0, h_ref = 1.0, R = 0.9;
        public double dt = 0.1; public int H = 200; public double gamma = 0.99;
        // command-level model (式27-29)
        public double tau_v = 0.25, tau_z = 0.30, tau_psi = 0.20;
        public double v_max_fb = 0.4, v_max_lr = 0.4, v_max_ud = 0.25;
        public double omega_max_psi = Math.PI / 3.0; // 60 deg/s
        public double wind_acc_std = 0.0, meas_pos_noise = 0.0;
        // formation (4節)
        public int caseId = 1;
        public double alpha0 = 0.0, v_eps = 0.05, lambda_phi = 0.3;
        public double v_scale = 0.3, cv = 2.0, alpha_min = 0.6, Kp_slot = 1.0;
        public string yaw_ref_mode = "look_target";
        // 観測正規化 (式82)
        public double R_max = 3.0, v_max_norm = 0.4, vo_max_norm = 0.3;
        public double vz_max_norm = 0.25, d_wall_max = 2.5, h_scale = 1.0;
        public int n_max_obs = -1;
        // 目的コスト重み (式93-94)
        public double w_slot = 5.0, w_R = 1.0, w_c = 1.0, w_e = 1.0, w_v = 0.2;
        public double w_u = 1e-3, w_du = 1e-2, w_h = 3.0, w_psi = 0.1;
        // 安全制約 (式103-104)
        public double d_drone = 0.5, d_target = 0.35, d_safe_wall = 0.35;
        public double e_h = 0.2, a_safe = 0.8;
        public int m_i = 5;
        public bool use_safety_shield = false;
        // 対象物 (式136,139,140)
        public string target_mode = "static";
        public double v_tar = 0.15, circle_A = 0.8, circle_omega = 0.2;
        public int poly_K_seg = 40;

        public int ResolvedNMaxObs() => n_max_obs < 0 ? (N - 1) : n_max_obs;

        // Python から来た config(JObject) を反映 (キー名は config.py と一致, case は予約語回避)
        public void ApplyJson(JObject c)
        {
            foreach (var prop in c.Properties())
            {
                string k = prop.Name;
                var v = prop.Value;
                switch (k)
                {
                    case "N": N = (int)v; break;
                    case "x_min": x_min = (double)v; break;
                    case "x_max": x_max = (double)v; break;
                    case "y_min": y_min = (double)v; break;
                    case "y_max": y_max = (double)v; break;
                    case "z_min": z_min = (double)v; break;
                    case "z_max": z_max = (double)v; break;
                    case "h_ref": h_ref = (double)v; break;
                    case "R": R = (double)v; break;
                    case "dt": dt = (double)v; break;
                    case "H": H = (int)v; break;
                    case "gamma": gamma = (double)v; break;
                    case "tau_v": tau_v = (double)v; break;
                    case "tau_z": tau_z = (double)v; break;
                    case "tau_psi": tau_psi = (double)v; break;
                    case "v_max_fb": v_max_fb = (double)v; break;
                    case "v_max_lr": v_max_lr = (double)v; break;
                    case "v_max_ud": v_max_ud = (double)v; break;
                    case "omega_max_psi": omega_max_psi = (double)v; break;
                    case "wind_acc_std": wind_acc_std = (double)v; break;
                    case "meas_pos_noise": meas_pos_noise = (double)v; break;
                    case "case": caseId = (int)v; break;
                    case "alpha0": alpha0 = (double)v; break;
                    case "v_eps": v_eps = (double)v; break;
                    case "lambda_phi": lambda_phi = (double)v; break;
                    case "v_scale": v_scale = (double)v; break;
                    case "cv": cv = (double)v; break;
                    case "alpha_min": alpha_min = (double)v; break;
                    case "Kp_slot": Kp_slot = (double)v; break;
                    case "yaw_ref_mode": yaw_ref_mode = (string)v; break;
                    case "R_max": R_max = (double)v; break;
                    case "v_max_norm": v_max_norm = (double)v; break;
                    case "vo_max_norm": vo_max_norm = (double)v; break;
                    case "vz_max_norm": vz_max_norm = (double)v; break;
                    case "d_wall_max": d_wall_max = (double)v; break;
                    case "h_scale": h_scale = (double)v; break;
                    case "n_max_obs": n_max_obs = (int)v; break;
                    case "w_slot": w_slot = (double)v; break;
                    case "w_R": w_R = (double)v; break;
                    case "w_c": w_c = (double)v; break;
                    case "w_e": w_e = (double)v; break;
                    case "w_v": w_v = (double)v; break;
                    case "w_u": w_u = (double)v; break;
                    case "w_du": w_du = (double)v; break;
                    case "w_h": w_h = (double)v; break;
                    case "w_psi": w_psi = (double)v; break;
                    case "d_drone": d_drone = (double)v; break;
                    case "d_target": d_target = (double)v; break;
                    case "d_safe_wall": d_safe_wall = (double)v; break;
                    case "e_h": e_h = (double)v; break;
                    case "a_safe": a_safe = (double)v; break;
                    case "m_i": m_i = (int)v; break;
                    case "use_safety_shield": use_safety_shield = (bool)v; break;
                    case "target_mode": target_mode = (string)v; break;
                    case "v_tar": v_tar = (double)v; break;
                    case "circle_A": circle_A = (double)v; break;
                    case "circle_omega": circle_omega = (double)v; break;
                    case "poly_K_seg": poly_K_seg = (int)v; break;
                    default: break; // d_i など未対応キーは無視
                }
            }
        }
    }
}
