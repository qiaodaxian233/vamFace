// VamFaceBridge v0.1 (initial draft)
// Virt-A-Mate 1.22.0.3 session plugin.
// Opens a TCP server on 127.0.0.1:8787 and accepts newline-delimited JSON
// commands, so an external MCP server (see /server) can drive VaM:
//   list atoms, read/write morphs, take screenshots, load scenes.
//
// Protocol: docs/protocol.md
//
// Design notes:
// - Socket accept/read happens on background threads; all VaM/Unity API
//   calls happen on the main thread (Update / coroutines). Requests are
//   queued and drained in Update().
// - Conservative C# on purpose (no string interpolation, no ?. etc.) —
//   VaM's runtime compiler is old Mono.
// - API names marked TODO(verify) are written from community-plugin
//   convention and MUST be validated in a live VaM once. Every handler
//   is wrapped in try/catch and returns a concrete error string instead
//   of failing silently or hard-gating (see 对话记忆 错误5).
//
// Install: copy this file to (VaM)/Custom/Scripts/VamFace/VamFaceBridge.cs
// then in VaM: Session Plugins -> Add Plugin -> select this file.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using SimpleJSON;

namespace VamFace
{
    public class VamFaceBridge : MVRScript
    {
        private const string VERSION = "0.1.0";
        private const int DEFAULT_PORT = 8787;
        private const int MAX_REQUESTS_PER_FRAME = 8;

        private TcpListener _listener;
        private Thread _acceptThread;
        private volatile bool _running;

        private readonly object _queueLock = new object();
        private readonly Queue<PendingRequest> _pending = new Queue<PendingRequest>();
        private readonly List<ClientConn> _clients = new List<ClientConn>();

        private JSONStorableString _statusJSON;
        private JSONStorableFloat _portJSON;

        // ------------------------------------------------------------------
        // Plumbing types
        // ------------------------------------------------------------------

        private class PendingRequest
        {
            public JSONClass json;
            public ClientConn conn;
        }

        private class ClientConn
        {
            public TcpClient client;
            public NetworkStream stream;
            public readonly object writeLock = new object();
            public volatile bool alive = true;

            public void SendLine(string line)
            {
                try
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(line + "\n");
                    lock (writeLock)
                    {
                        stream.Write(bytes, 0, bytes.Length);
                        stream.Flush();
                    }
                }
                catch (Exception)
                {
                    alive = false;
                }
            }
        }

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------

        public override void Init()
        {
            try
            {
                _statusJSON = new JSONStorableString("Status", "stopped");
                UIDynamicTextField statusField = CreateTextField(_statusJSON);
                statusField.height = 80f;

                _portJSON = new JSONStorableFloat("Port", DEFAULT_PORT, 1024f, 65535f, true, true);
                RegisterFloat(_portJSON);
                CreateSlider(_portJSON);

                StartServer((int)_portJSON.val);
            }
            catch (Exception e)
            {
                SuperController.LogError("VamFaceBridge Init failed: " + e);
            }
        }

        private void StartServer(int port)
        {
            StopServer();
            try
            {
                _listener = new TcpListener(IPAddress.Loopback, port);
                _listener.Start();
                _running = true;
                _acceptThread = new Thread(AcceptLoop);
                _acceptThread.IsBackground = true;
                _acceptThread.Start();
                SetStatus("listening on 127.0.0.1:" + port);
                SuperController.LogMessage("VamFaceBridge v" + VERSION + " listening on 127.0.0.1:" + port);
            }
            catch (Exception e)
            {
                SetStatus("failed to start: " + e.Message);
                SuperController.LogError("VamFaceBridge failed to start: " + e);
            }
        }

        private void StopServer()
        {
            _running = false;
            try { if (_listener != null) _listener.Stop(); } catch (Exception) { }
            _listener = null;
            lock (_clients)
            {
                for (int i = 0; i < _clients.Count; i++)
                {
                    try { _clients[i].client.Close(); } catch (Exception) { }
                }
                _clients.Clear();
            }
        }

        private void OnDestroy()
        {
            StopServer();
        }

        private void SetStatus(string s)
        {
            if (_statusJSON != null) _statusJSON.val = s;
        }

        // ------------------------------------------------------------------
        // Network threads
        // ------------------------------------------------------------------

        private void AcceptLoop()
        {
            while (_running)
            {
                try
                {
                    TcpClient client = _listener.AcceptTcpClient();
                    ClientConn conn = new ClientConn();
                    conn.client = client;
                    conn.stream = client.GetStream();
                    lock (_clients) { _clients.Add(conn); }

                    Thread reader = new Thread(delegate () { ReadLoop(conn); });
                    reader.IsBackground = true;
                    reader.Start();
                }
                catch (Exception)
                {
                    if (_running) Thread.Sleep(200);
                }
            }
        }

        private void ReadLoop(ClientConn conn)
        {
            StreamReader reader = new StreamReader(conn.stream, Encoding.UTF8);
            try
            {
                while (_running && conn.alive)
                {
                    string line = reader.ReadLine();
                    if (line == null) break;
                    line = line.Trim();
                    if (line.Length == 0) continue;

                    JSONClass json = null;
                    try { json = JSON.Parse(line) as JSONClass; }
                    catch (Exception) { }

                    if (json == null)
                    {
                        conn.SendLine("{\"ok\":false,\"error\":\"invalid json\"}");
                        continue;
                    }

                    PendingRequest req = new PendingRequest();
                    req.json = json;
                    req.conn = conn;
                    lock (_queueLock) { _pending.Enqueue(req); }
                }
            }
            catch (Exception) { }
            finally
            {
                conn.alive = false;
                try { conn.client.Close(); } catch (Exception) { }
                lock (_clients) { _clients.Remove(conn); }
            }
        }

        // ------------------------------------------------------------------
        // Main-thread dispatch
        // ------------------------------------------------------------------

        private void Update()
        {
            int budget = MAX_REQUESTS_PER_FRAME;
            while (budget > 0)
            {
                PendingRequest req = null;
                lock (_queueLock)
                {
                    if (_pending.Count == 0) break;
                    req = _pending.Dequeue();
                }
                budget--;
                HandleRequest(req);
            }
        }

        private void HandleRequest(PendingRequest req)
        {
            string id = req.json["id"] != null ? req.json["id"].Value : "";
            string cmd = req.json["cmd"] != null ? req.json["cmd"].Value : "";
            JSONClass args = req.json["args"] as JSONClass;
            if (args == null) args = new JSONClass();

            try
            {
                switch (cmd)
                {
                    case "ping": Reply(req.conn, id, CmdPing()); break;
                    case "list_atoms": Reply(req.conn, id, CmdListAtoms()); break;
                    case "list_morphs": Reply(req.conn, id, CmdListMorphs(args)); break;
                    case "get_morphs": Reply(req.conn, id, CmdGetMorphs(args)); break;
                    case "set_morphs": Reply(req.conn, id, CmdSetMorphs(args)); break;
                    case "reset_morphs": Reply(req.conn, id, CmdResetMorphs(args)); break;
                    case "load_scene": Reply(req.conn, id, CmdLoadScene(args)); break;
                    case "focus_head": Reply(req.conn, id, CmdFocusHead(args)); break;
                    case "screenshot":
                        // async: coroutine replies when the frame is captured
                        StartCoroutine(CaptureRoutine(req.conn, id, args));
                        break;
                    default:
                        ReplyError(req.conn, id, "unknown cmd: " + cmd);
                        break;
                }
            }
            catch (Exception e)
            {
                ReplyError(req.conn, id, cmd + " failed: " + e.Message);
            }
        }

        private void Reply(ClientConn conn, string id, JSONClass data)
        {
            JSONClass resp = new JSONClass();
            resp["id"] = id;
            resp["ok"].AsBool = true;
            resp["data"] = data;
            conn.SendLine(resp.ToString());
        }

        private void ReplyError(ClientConn conn, string id, string error)
        {
            JSONClass resp = new JSONClass();
            resp["id"] = id;
            resp["ok"].AsBool = false;
            resp["error"] = error;
            conn.SendLine(resp.ToString());
        }

        // ------------------------------------------------------------------
        // Command handlers
        // ------------------------------------------------------------------

        private JSONClass CmdPing()
        {
            JSONClass d = new JSONClass();
            d["version"] = VERSION;
            d["app"] = "VaM";
            return d;
        }

        private JSONClass CmdListAtoms()
        {
            JSONClass d = new JSONClass();
            JSONArray arr = new JSONArray();
            List<Atom> atoms = SuperController.singleton.GetAtoms();
            for (int i = 0; i < atoms.Count; i++)
            {
                JSONClass a = new JSONClass();
                a["uid"] = atoms[i].uid;
                a["type"] = atoms[i].type;
                arr.Add(a);
            }
            d["atoms"] = arr;
            return d;
        }

        // Resolve a Person atom's morph control UI.
        // TODO(verify): DAZCharacterSelector / morphsControlUI member names
        // against VaM 1.22 — these follow common community-plugin usage.
        private GenerateDAZMorphsControlUI GetMorphsControl(string atomUid, out string error)
        {
            error = null;
            Atom atom = SuperController.singleton.GetAtomByUid(atomUid);
            if (atom == null)
            {
                error = "atom not found: " + atomUid;
                return null;
            }
            DAZCharacterSelector selector = atom.GetComponentInChildren<DAZCharacterSelector>();
            if (selector == null)
            {
                error = "atom is not a Person (no DAZCharacterSelector): " + atomUid;
                return null;
            }
            GenerateDAZMorphsControlUI ui = selector.morphsControlUI;
            if (ui == null)
            {
                error = "morphsControlUI unavailable on: " + atomUid;
                return null;
            }
            return ui;
        }

        private DAZMorph FindMorph(GenerateDAZMorphsControlUI ui, string name)
        {
            DAZMorph m = ui.GetMorphByDisplayName(name);
            if (m == null) m = ui.GetMorphByUid(name); // TODO(verify) method name
            return m;
        }

        private JSONClass CmdListMorphs(JSONClass args)
        {
            string atomUid = args["atom"].Value;
            string filter = args["filter"] != null ? args["filter"].Value.ToLowerInvariant() : "";
            string region = args["region"] != null ? args["region"].Value.ToLowerInvariant() : "";
            int limit = args["limit"] != null ? args["limit"].AsInt : 200;
            if (limit <= 0) limit = 200;

            string error;
            GenerateDAZMorphsControlUI ui = GetMorphsControl(atomUid, out error);
            if (ui == null) throw new Exception(error);

            JSONArray arr = new JSONArray();
            int count = 0;
            List<DAZMorph> morphs = ui.GetMorphs(); // TODO(verify) method name
            for (int i = 0; i < morphs.Count && count < limit; i++)
            {
                DAZMorph m = morphs[i];
                string dn = m.displayName != null ? m.displayName : "";
                string rg = m.region != null ? m.region : "";
                if (filter.Length > 0 && dn.ToLowerInvariant().IndexOf(filter) < 0) continue;
                if (region.Length > 0 && rg.ToLowerInvariant().IndexOf(region) < 0) continue;

                JSONClass e = new JSONClass();
                e["name"] = dn;
                e["uid"] = m.uid;
                e["region"] = rg;
                e["value"].AsFloat = m.morphValue;
                e["min"].AsFloat = m.min;
                e["max"].AsFloat = m.max;
                arr.Add(e);
                count++;
            }

            JSONClass d = new JSONClass();
            d["count"].AsInt = count;
            d["total"].AsInt = morphs.Count;
            d["morphs"] = arr;
            return d;
        }

        private JSONClass CmdGetMorphs(JSONClass args)
        {
            string atomUid = args["atom"].Value;
            bool changedOnly = args["changed_only"] == null || args["changed_only"].AsBool;

            string error;
            GenerateDAZMorphsControlUI ui = GetMorphsControl(atomUid, out error);
            if (ui == null) throw new Exception(error);

            JSONClass values = new JSONClass();
            List<DAZMorph> morphs = ui.GetMorphs();
            for (int i = 0; i < morphs.Count; i++)
            {
                DAZMorph m = morphs[i];
                float def = m.startValue; // TODO(verify) default-value field name
                if (changedOnly && Mathf.Abs(m.morphValue - def) < 0.0001f) continue;
                values[m.displayName].AsFloat = m.morphValue;
            }

            JSONClass d = new JSONClass();
            d["values"] = values;
            return d;
        }

        private JSONClass CmdSetMorphs(JSONClass args)
        {
            string atomUid = args["atom"].Value;
            JSONClass values = args["values"] as JSONClass;
            bool clamp = args["clamp"] == null || args["clamp"].AsBool;
            if (values == null) throw new Exception("set_morphs requires args.values object");

            string error;
            GenerateDAZMorphsControlUI ui = GetMorphsControl(atomUid, out error);
            if (ui == null) throw new Exception(error);

            int applied = 0;
            JSONArray missing = new JSONArray();
            foreach (KeyValuePair<string, JSONNode> kv in values)
            {
                DAZMorph m = FindMorph(ui, kv.Key);
                if (m == null)
                {
                    missing.Add(new JSONData(kv.Key));
                    continue;
                }
                float v = kv.Value.AsFloat;
                if (clamp) v = Mathf.Clamp(v, m.min, m.max);
                m.morphValue = v;
                applied++;
            }

            JSONClass d = new JSONClass();
            d["applied"].AsInt = applied;
            d["missing"] = missing;
            return d;
        }

        private JSONClass CmdResetMorphs(JSONClass args)
        {
            string atomUid = args["atom"].Value;
            string error;
            GenerateDAZMorphsControlUI ui = GetMorphsControl(atomUid, out error);
            if (ui == null) throw new Exception(error);

            int resetCount = 0;
            List<DAZMorph> morphs = ui.GetMorphs();
            for (int i = 0; i < morphs.Count; i++)
            {
                DAZMorph m = morphs[i];
                if (Mathf.Abs(m.morphValue - m.startValue) < 0.0001f) continue;
                m.morphValue = m.startValue;
                resetCount++;
            }
            JSONClass d = new JSONClass();
            d["reset"].AsInt = resetCount;
            return d;
        }

        private JSONClass CmdLoadScene(JSONClass args)
        {
            string path = args["path"].Value;
            if (path == null || path.Length == 0) throw new Exception("load_scene requires args.path");
            SuperController.singleton.Load(path); // TODO(verify) exact overload for .json scene path
            JSONClass d = new JSONClass();
            d["loading"] = path;
            return d;
        }

        // Point the window camera at the head controller so screenshots frame
        // the face consistently for the fitting loop.
        private JSONClass CmdFocusHead(JSONClass args)
        {
            string atomUid = args["atom"].Value;
            Atom atom = SuperController.singleton.GetAtomByUid(atomUid);
            if (atom == null) throw new Exception("atom not found: " + atomUid);

            FreeControllerV3 head = null;
            FreeControllerV3[] controllers = atom.freeControllers;
            for (int i = 0; i < controllers.Length; i++)
            {
                if (controllers[i].name == "headControl") { head = controllers[i]; break; }
            }
            if (head == null) throw new Exception("headControl not found on: " + atomUid);

            SuperController.singleton.FocusOnController(head); // TODO(verify) signature in 1.22
            JSONClass d = new JSONClass();
            d["focused"] = atomUid + "/headControl";
            return d;
        }

        // ------------------------------------------------------------------
        // Screenshot (async, replies from coroutine)
        // ------------------------------------------------------------------

        private IEnumerator CaptureRoutine(ClientConn conn, string id, JSONClass args)
        {
            int maxWidth = args["max_width"] != null ? args["max_width"].AsInt : 0;

            yield return new WaitForEndOfFrame();

            byte[] png = null;
            string error = null;
            int outW = 0, outH = 0;
            try
            {
                int w = Screen.width;
                int h = Screen.height;
                Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
                tex.Apply();

                if (maxWidth > 0 && w > maxWidth)
                {
                    int nw = maxWidth;
                    int nh = Mathf.RoundToInt((float)h * nw / w);
                    RenderTexture rt = RenderTexture.GetTemporary(nw, nh);
                    Graphics.Blit(tex, rt);
                    RenderTexture prev = RenderTexture.active;
                    RenderTexture.active = rt;
                    Texture2D small = new Texture2D(nw, nh, TextureFormat.RGB24, false);
                    small.ReadPixels(new Rect(0, 0, nw, nh), 0, 0);
                    small.Apply();
                    RenderTexture.active = prev;
                    RenderTexture.ReleaseTemporary(rt);
                    UnityEngine.Object.Destroy(tex);
                    tex = small;
                    w = nw; h = nh;
                }

                png = tex.EncodeToPNG();
                outW = w; outH = h;
                UnityEngine.Object.Destroy(tex);
            }
            catch (Exception e)
            {
                error = "screenshot failed: " + e.Message;
            }

            if (error != null)
            {
                ReplyError(conn, id, error);
            }
            else
            {
                JSONClass d = new JSONClass();
                d["width"].AsInt = outW;
                d["height"].AsInt = outH;
                d["png_base64"] = Convert.ToBase64String(png);
                Reply(conn, id, d);
            }
        }
    }
}
