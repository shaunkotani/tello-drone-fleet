// Logger.cs
// 9.5 ログ出力。各ステップの目的コスト・制約コスト・取り囲み誤差を CSV に保存する。
// Python 側でも train.py が CSV を出すので、Unity 側は可視化/デバッグ補助用。
using System.IO;
using System.Text;

namespace TelloEnclosing
{
    public class Logger
    {
        private StreamWriter writer;

        public Logger(string path)
        {
            writer = new StreamWriter(path, false);
            writer.WriteLine("k,agent,obj,c1,c2,c3,c4,c5,slot_err");
        }

        public void Log(StepResult sr)
        {
            int N = sr.objectiveCosts.Length;
            for (int i = 0; i < N; i++)
            {
                double slotErr = (sr.r[i] - sr.slots[i]).Norm();
                var c = sr.constraintCosts[i];
                var sb = new StringBuilder();
                sb.Append(sr.k).Append(',').Append(i).Append(',')
                  .Append(sr.objectiveCosts[i].ToString("F5")).Append(',')
                  .Append(c[0].ToString("F5")).Append(',').Append(c[1].ToString("F5")).Append(',')
                  .Append(c[2].ToString("F5")).Append(',').Append(c[3].ToString("F5")).Append(',')
                  .Append(c[4].ToString("F5")).Append(',').Append(slotErr.ToString("F5"));
                writer.WriteLine(sb.ToString());
            }
            writer.Flush();
        }

        public void Close() { writer?.Close(); }
    }
}
