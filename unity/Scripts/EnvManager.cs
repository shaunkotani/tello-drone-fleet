// EnvManager.cs
// 9.3 Algorithm 2 (Unity 1 ステップ処理) のオーケストレーション。
// python/tello_rl/mock_env.py と同一の計算順序でミラー。
// raw action -> 安全シールド -> ダイナミクス前進 -> 対象物/スロット更新
// -> 目的・制約コスト評価 -> 観測構成 -> 終了判定。
using System;

namespace TelloEnclosing
{
    public class StepResult
    {
        public double[][] obs;            // (N, od)
        public double[] objectiveCosts;   // (N,)
        public double[][] constraintCosts;// (N, 5)
        public double[][] executed;       // (N, 4)
        public bool done;
        public string reason;
        public int k;
        public V2 ro;
        public V2[] r, slots;
        public double[] edgeRef;
        public double phi;
    }

    public class EnvManager
    {
        public EnvConfig cfg;
        public TelloAgent[] agents;
        private TargetObject target;
        private FormationManager fm;
        private double[][] aPrev;
        private double[] battery;
        private int k;
        private System.Random rng;

        public EnvManager(EnvConfig cfg, int seed = 0)
        {
            this.cfg = cfg;
            rng = new System.Random(seed);
            target = new TargetObject(cfg);
            fm = new FormationManager(cfg);
            ResetInternal();
        }

        private void ResetInternal()
        {
            k = 0;
            target.Reset();
            fm.phi = 0.0;
            fm.ComputeSlots(target.ro, target.vo);
            int N = cfg.N;
            agents = new TelloAgent[N];
            aPrev = new double[N][];
            battery = new double[N];
            for (int i = 0; i < N; i++)
            {
                V2 r0 = fm.slots[i] + new V2(0.05 * Gauss(), 0.05 * Gauss());
                V2 d = target.ro - r0;
                double psi0 = Math.Atan2(d.y, d.x);
                agents[i] = new TelloAgent(r0, psi0, cfg.h_ref);
                aPrev[i] = new double[4];
                battery[i] = 100.0;
            }
        }

        public double[][] Reset()
        {
            ResetInternal();
            return AllObs();
        }

        private double[][] AllObs()
        {
            int N = cfg.N;
            var o = new double[N][];
            for (int i = 0; i < N; i++)
                o[i] = ObservationBuilder.Build(i, agents, target.ro, target.vo,
                                                fm.slots, aPrev[i], battery[i], cfg);
            return o;
        }

        // 安全シールド S_i (数値実験では恒等写像 = clip(raw))
        private double[] SafetyShield(int i, double[] raw)
        {
            double[] c = new double[4];
            for (int q = 0; q < 4; q++) c[q] = Math.Max(-1.0, Math.Min(1.0, raw[q]));
            // use_safety_shield=true なら command-level 安全フィルタをここに実装
            return c;
        }

        public StepResult Step(double[][] rawActions)
        {
            int N = cfg.N;
            // 1. clip + 安全シールド
            var a = new double[N][];
            for (int i = 0; i < N; i++) a[i] = SafetyShield(i, rawActions[i]);

            // 2. 目的コスト用 世界座標速度指令 (現時刻 psi)
            var vxyCmd = new V2[N];
            var rPrev = new V2[N];
            for (int i = 0; i < N; i++)
            {
                vxyCmd[i] = TelloAgent.ActionToWorldVelCmd(a[i], agents[i].psi, cfg);
                rPrev[i] = agents[i].r;
            }

            // 3. ダイナミクス前進 (式21-26)
            for (int i = 0; i < N; i++) agents[i].Step(a[i], cfg, rng);

            // 4. 対象物更新
            var slotsPrev = fm.slots;
            target.Step();

            // 5. 進行方向角・スロット更新
            fm.UpdateHeading(target.vo);
            fm.ComputeSlots(target.ro, target.vo);

            // 6. 参照速度 (式53,91)
            var vref = SafetyEvaluator.ReferenceVelocity(slotsPrev, fm.slots, rPrev, cfg);
            var vrefClip = new V2[N];
            for (int i = 0; i < N; i++) vrefClip[i] = SafetyEvaluator.ClipReference(vref[i], cfg);

            var rNext = new V2[N];
            for (int i = 0; i < N; i++) rNext[i] = agents[i].r;

            // 7. 目的・制約コスト
            var obj = new double[N];
            var con = new double[N][];
            for (int i = 0; i < N; i++)
            {
                double psiRef = SafetyEvaluator.YawReference(i, rNext, target.ro, fm.phi, cfg);
                obj[i] = SafetyEvaluator.ObjectiveCost(
                    i, rNext, target.ro, fm.slots, fm.rCgRef, fm.edgeRef,
                    vxyCmd[i], vrefClip[i], a[i], aPrev[i],
                    agents[i].h, agents[i].psi, psiRef, cfg);
                con[i] = SafetyEvaluator.ConstraintCost(i, rNext, target.ro, agents[i].h, a[i], cfg);
            }

            // 8. 観測・前回入力更新
            for (int i = 0; i < N; i++) aPrev[i] = (double[])a[i].Clone();
            k += 1;
            var obs = AllObs();

            // 9. 終了判定
            string reason; bool done = CheckDone(rNext, out reason);

            return new StepResult
            {
                obs = obs, objectiveCosts = obj, constraintCosts = con,
                executed = a, done = done, reason = reason, k = k,
                ro = target.ro, r = rNext, slots = fm.slots,
                edgeRef = fm.edgeRef, phi = fm.phi
            };
        }

        private bool CheckDone(V2[] r, out string reason)
        {
            for (int i = 0; i < cfg.N; i++)
            {
                if (!(cfg.x_min <= r[i].x && r[i].x <= cfg.x_max &&
                      cfg.y_min <= r[i].y && r[i].y <= cfg.y_max)) { reason = "region"; return true; }
                if (!(cfg.z_min <= agents[i].h && agents[i].h <= cfg.z_max)) { reason = "altitude"; return true; }
            }
            for (int i = 0; i < cfg.N; i++)
                for (int j = i + 1; j < cfg.N; j++)
                    if ((r[i] - r[j]).Norm() < 0.15) { reason = "collision"; return true; }
            if (k >= cfg.H) { reason = "time"; return true; }
            reason = ""; return false;
        }

        private double Gauss()
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
