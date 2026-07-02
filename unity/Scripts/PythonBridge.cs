// PythonBridge.cs
// 9.1 Python<->Unity インターフェース。Unity をサーバとして TCP listen し、
// Python クライアント(unity_env.UnityEnv)からの config/reset/step/close を処理する。
// EnvManager は Unity API を呼ばない純計算なのでバックグラウンドスレッドで実行可能。
// 可視化したい場合はメインスレッドから EnvManager の状態を読み出して描画する。
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json.Linq;

namespace TelloEnclosing
{
    public class PythonBridge : MonoBehaviour
    {
        public int port = 5005;
        public int seed = 0;

        private TcpListener listener;
        private Thread serverThread;
        private volatile bool running;
        private EnvConfig cfg = new EnvConfig();
        private EnvManager env;
        public EnvManager Env => env;   // 可視化用に公開

        void Start()
        {
            env = new EnvManager(cfg, seed);
            running = true;
            serverThread = new Thread(ServerLoop) { IsBackground = true };
            serverThread.Start();
            Debug.Log($"[PythonBridge] listening on port {port}");
        }

        void OnDestroy() { Stop(); }
        void OnApplicationQuit() { Stop(); }

        private void Stop()
        {
            running = false;
            try { listener?.Stop(); } catch { }
            try { serverThread?.Join(200); } catch { }
        }

        private void ServerLoop()
        {
            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start();
            while (running)
            {
                TcpClient client = null;
                try { client = listener.AcceptTcpClient(); }
                catch { break; }
                using (client)
                using (var stream = client.GetStream())
                {
                    try { HandleClient(stream); }
                    catch (Exception e) { Debug.LogWarning($"[PythonBridge] {e.Message}"); }
                }
            }
        }

        private void HandleClient(NetworkStream stream)
        {
            while (running)
            {
                JObject msg = Protocol.RecvMsg(stream);
                string cmd = (string)msg["cmd"];
                if (cmd == "config")
                {
                    cfg.ApplyJson((JObject)msg["config"]);
                    env = new EnvManager(cfg, seed);
                    Protocol.SendMsg(stream, new JObject { ["ok"] = true });
                }
                else if (cmd == "reset")
                {
                    double[][] obs = env.Reset();
                    var rep = new JObject { ["obs"] = ToJArray2D(obs), ["info"] = new JObject() };
                    Protocol.SendMsg(stream, rep);
                }
                else if (cmd == "step")
                {
                    double[][] raw = ToDouble2D((JArray)msg["raw_actions"], cfg.N, 4);
                    StepResult sr = env.Step(raw);
                    Protocol.SendMsg(stream, BuildStepReply(sr));
                }
                else if (cmd == "close") { break; }
            }
        }

        private JObject BuildStepReply(StepResult sr)
        {
            var info = new JObject
            {
                ["reason"] = sr.reason,
                ["k"] = sr.k,
                ["ro"] = ToJArray1D(sr.ro.ToArray()),
                ["r"] = ToJArrayV2(sr.r),
                ["slots"] = ToJArrayV2(sr.slots),
                ["edge_ref"] = ToJArray1D(sr.edgeRef),
                ["phi"] = sr.phi
            };
            return new JObject
            {
                ["obs"] = ToJArray2D(sr.obs),
                ["objective_costs"] = ToJArray1D(sr.objectiveCosts),
                ["constraint_costs"] = ToJArray2D(sr.constraintCosts),
                ["executed_actions"] = ToJArray2D(sr.executed),
                ["done"] = sr.done,
                ["info"] = info
            };
        }

        // --- JSON 変換ヘルパ ---
        private static JArray ToJArray1D(double[] a)
        {
            var arr = new JArray(); foreach (var x in a) arr.Add(x); return arr;
        }
        private static JArray ToJArray2D(double[][] a)
        {
            var outer = new JArray();
            foreach (var row in a) outer.Add(ToJArray1D(row));
            return outer;
        }
        private static JArray ToJArrayV2(V2[] a)
        {
            var outer = new JArray();
            foreach (var v in a) outer.Add(ToJArray1D(v.ToArray()));
            return outer;
        }
        private static double[][] ToDouble2D(JArray a, int N, int M)
        {
            var outArr = new double[N][];
            for (int i = 0; i < N; i++)
            {
                var row = (JArray)a[i];
                outArr[i] = new double[M];
                for (int j = 0; j < M; j++) outArr[i][j] = (double)row[j];
            }
            return outArr;
        }
    }
}
