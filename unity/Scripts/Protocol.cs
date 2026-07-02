// Protocol.cs
// Python <-> Unity インターフェース (9.1)。
// 長さ前置き (4byte big-endian) + UTF-8 JSON。python/tello_rl/protocol.py と一致させること。
// JSON は Newtonsoft.Json (Unity Package Manager: com.unity.nuget.newtonsoft-json) を使用。
using System;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json.Linq;

namespace TelloEnclosing
{
    public static class Protocol
    {
        // 4byte big-endian 長さ前置きで JSON 文字列を送信
        public static void SendMsg(NetworkStream stream, JObject obj)
        {
            byte[] payload = Encoding.UTF8.GetBytes(obj.ToString(Newtonsoft.Json.Formatting.None));
            byte[] header = BitConverter.GetBytes(payload.Length);
            if (BitConverter.IsLittleEndian) Array.Reverse(header); // big-endian
            stream.Write(header, 0, 4);
            stream.Write(payload, 0, payload.Length);
            stream.Flush();
        }

        public static JObject RecvMsg(NetworkStream stream)
        {
            byte[] header = RecvExact(stream, 4);
            if (BitConverter.IsLittleEndian) Array.Reverse(header);
            int length = BitConverter.ToInt32(header, 0);
            byte[] data = RecvExact(stream, length);
            return JObject.Parse(Encoding.UTF8.GetString(data));
        }

        private static byte[] RecvExact(NetworkStream stream, int n)
        {
            byte[] buf = new byte[n];
            int off = 0;
            while (off < n)
            {
                int read = stream.Read(buf, off, n - off);
                if (read <= 0) throw new Exception("socket closed");
                off += read;
            }
            return buf;
        }
    }
}
