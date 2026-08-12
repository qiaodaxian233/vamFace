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
// NOTE(sandbox, verified live 2026-08-12 on VaM 1.22.0.3):
// VaM's plugin sandbox scans the compiled assembly and PROHIBITS the whole
// System.IO namespace (SecurityException: NamespaceRestriction). That bans
// NetworkStream-as-Stream usage, StreamReader/Writer, MemoryStream, File,
// Path... Networking below therefore uses the raw Socket byte[] API
// (System.Net.Sockets only) with manual newline framing.
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
        private const string VERSION = "0.5.5";
        // 与 server 端 vamface_mcp.PROTOCOL_VERSION 对账,改协议时两边同步 +1。
        private const int PROTOCOL = 1;
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
            public Socket socket;
            public readonly object writeLock = new object();
            public volatile bool alive = true;
            // Receive-side accumulator for newline framing. Only touched by
            // this connection's reader thread, so no lock needed.
            public byte[] acc = new byte[4096];
            public int accLen = 0;

            public void SendLine(string line)
            {
                try
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(line + "\n");
                    lock (writeLock)
                    {
                        int sent = 0;
                        while (sent < bytes.Length)
                        {
                            int n = socket.Send(bytes, sent, bytes.Length - sent,
                                                SocketFlags.None);
                            if (n <= 0) throw new SocketException();
                            sent += n;
                        }
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
                    try { _clients[i].socket.Close(); } catch (Exception) { }
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
                    Socket sock = _listener.AcceptSocket();
                    sock.NoDelay = true;  // request/reply JSON lines - don't batch
                    ClientConn conn = new ClientConn();
                    conn.socket = sock;
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
            byte[] buf = new byte[8192];
            try
            {
                while (_running && conn.alive)
                {
                    int n = conn.socket.Receive(buf, 0, buf.Length, SocketFlags.None);
                    if (n <= 0) break;

                    // Grow the accumulator if needed.
                    if (conn.accLen + n > conn.acc.Length)
                    {
                        int cap = conn.acc.Length * 2;
                        while (cap < conn.accLen + n) cap *= 2;
                        byte[] bigger = new byte[cap];
                        Array.Copy(conn.acc, 0, bigger, 0, conn.accLen);
                        conn.acc = bigger;
                    }
                    Array.Copy(buf, 0, conn.acc, conn.accLen, n);
                    conn.accLen += n;

                    // Extract complete lines. '\n' (0x0A) never occurs inside
                    // a UTF-8 multibyte sequence, so byte-level scan is safe.
                    int start = 0;
                    for (int i = 0; i < conn.accLen; i++)
                    {
                        if (conn.acc[i] != (byte)'\n') continue;
                        int len = i - start;
                        if (len > 0 && conn.acc[start + len - 1] == (byte)'\r') len--;
                        if (len > 0)
                        {
                            string line = Encoding.UTF8.GetString(conn.acc, start, len).Trim();
                            if (line.Length > 0) HandleLine(conn, line);
                        }
                        start = i + 1;
                    }
                    if (start > 0)
                    {
                        int remain = conn.accLen - start;
                        if (remain > 0) Array.Copy(conn.acc, start, conn.acc, 0, remain);
                        conn.accLen = remain;
                    }
                }
            }
            catch (Exception) { }
            finally
            {
                conn.alive = false;
                try { conn.socket.Close(); } catch (Exception) { }
                lock (_clients) { _clients.Remove(conn); }
            }
        }

        private void HandleLine(ClientConn conn, string line)
        {
            JSONClass json = null;
            try { json = JSON.Parse(line) as JSONClass; }
            catch (Exception) { }

            if (json == null)
            {
                conn.SendLine("{\"ok\":false,\"error\":\"invalid json\"}");
                return;
            }

            PendingRequest req = new PendingRequest();
            req.json = json;
            req.conn = conn;
            lock (_queueLock) { _pending.Enqueue(req); }
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
                    case "list_storables": Reply(req.conn, id, CmdListStorables(args)); break;
                    case "list_params": Reply(req.conn, id, CmdListParams(args)); break;
                    case "get_param": Reply(req.conn, id, CmdGetParam(args)); break;
                    case "set_param": Reply(req.conn, id, CmdSetParam(args)); break;
                    case "call_action": Reply(req.conn, id, CmdCallAction(args)); break;
                    case "list_characters": Reply(req.conn, id, CmdListCharacters(args)); break;
                    case "set_character": Reply(req.conn, id, CmdSetCharacter(args)); break;
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
            d["protocol"].AsInt = PROTOCOL;
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
        // Generic storable/param access
        // Lets the client DISCOVER real parameter names (e.g. skin color
        // params) at runtime instead of us hard-coding guesses.
        // ------------------------------------------------------------------

        private Atom RequireAtom(string uid)
        {
            Atom atom = SuperController.singleton.GetAtomByUid(uid);
            if (atom == null) throw new Exception("atom not found: " + uid);
            return atom;
        }

        private JSONStorable RequireStorable(string atomUid, string storableId)
        {
            Atom atom = RequireAtom(atomUid);
            JSONStorable st = atom.GetStorableByID(storableId);
            if (st == null) throw new Exception("storable not found: " + atomUid + "/" + storableId);
            return st;
        }

        private JSONClass CmdListStorables(JSONClass args)
        {
            Atom atom = RequireAtom(args["atom"].Value);
            JSONArray arr = new JSONArray();
            List<string> ids = atom.GetStorableIDs();
            for (int i = 0; i < ids.Count; i++) arr.Add(new JSONData(ids[i]));
            JSONClass d = new JSONClass();
            d["storables"] = arr;
            return d;
        }

        private static void AddNameArray(JSONClass d, string key, List<string> names)
        {
            JSONArray arr = new JSONArray();
            if (names != null)
            {
                for (int i = 0; i < names.Count; i++) arr.Add(new JSONData(names[i]));
            }
            d[key] = arr;
        }

        private JSONClass CmdListParams(JSONClass args)
        {
            JSONStorable st = RequireStorable(args["atom"].Value, args["storable"].Value);
            JSONClass d = new JSONClass();
            // TODO(verify): these accessor names against 1.22
            AddNameArray(d, "floats", st.GetFloatParamNames());
            AddNameArray(d, "bools", st.GetBoolParamNames());
            AddNameArray(d, "colors", st.GetColorParamNames());
            AddNameArray(d, "choosers", st.GetStringChooserParamNames());
            AddNameArray(d, "strings", st.GetStringParamNames());
            return d;
        }

        private JSONClass CmdGetParam(JSONClass args)
        {
            JSONStorable st = RequireStorable(args["atom"].Value, args["storable"].Value);
            string name = args["param"].Value;
            JSONClass d = new JSONClass();

            JSONStorableFloat jf = st.GetFloatJSONParam(name);
            if (jf != null)
            {
                d["type"] = "float";
                d["value"].AsFloat = jf.val;
                d["min"].AsFloat = jf.min;
                d["max"].AsFloat = jf.max;
                return d;
            }
            JSONStorableBool jb = st.GetBoolJSONParam(name);
            if (jb != null)
            {
                d["type"] = "bool";
                d["value"].AsBool = jb.val;
                return d;
            }
            JSONStorableColor jc = st.GetColorJSONParam(name);
            if (jc != null)
            {
                d["type"] = "color";
                JSONClass hsv = new JSONClass();
                hsv["h"].AsFloat = jc.val.H; // TODO(verify) HSVColor field names
                hsv["s"].AsFloat = jc.val.S;
                hsv["v"].AsFloat = jc.val.V;
                d["value"] = hsv;
                return d;
            }
            JSONStorableStringChooser jch = st.GetStringChooserJSONParam(name);
            if (jch != null)
            {
                d["type"] = "chooser";
                d["value"] = jch.val;
                JSONArray choices = new JSONArray();
                if (jch.choices != null)
                {
                    for (int i = 0; i < jch.choices.Count; i++) choices.Add(new JSONData(jch.choices[i]));
                }
                d["choices"] = choices;
                return d;
            }
            JSONStorableString js = st.GetStringJSONParam(name);
            if (js != null)
            {
                d["type"] = "string";
                d["value"] = js.val;
                return d;
            }
            throw new Exception("param not found: " + name);
        }

        private JSONClass CmdSetParam(JSONClass args)
        {
            JSONStorable st = RequireStorable(args["atom"].Value, args["storable"].Value);
            string name = args["param"].Value;
            string type = args["type"] != null ? args["type"].Value : "";
            JSONNode value = args["value"];
            if (value == null) throw new Exception("set_param requires args.value");
            JSONClass d = new JSONClass();

            if (type == "" || type == "float")
            {
                JSONStorableFloat jf = st.GetFloatJSONParam(name);
                if (jf != null)
                {
                    jf.val = value.AsFloat;
                    d["type"] = "float";
                    d["value"].AsFloat = jf.val;
                    return d;
                }
                if (type == "float") throw new Exception("float param not found: " + name);
            }
            if (type == "" || type == "bool")
            {
                JSONStorableBool jb = st.GetBoolJSONParam(name);
                if (jb != null)
                {
                    jb.val = value.AsBool;
                    d["type"] = "bool";
                    d["value"].AsBool = jb.val;
                    return d;
                }
                if (type == "bool") throw new Exception("bool param not found: " + name);
            }
            if (type == "" || type == "color")
            {
                JSONStorableColor jc = st.GetColorJSONParam(name);
                if (jc != null)
                {
                    HSVColor c = new HSVColor();
                    c.H = value["h"].AsFloat; // TODO(verify) HSVColor field names
                    c.S = value["s"].AsFloat;
                    c.V = value["v"].AsFloat;
                    jc.val = c;
                    d["type"] = "color";
                    return d;
                }
                if (type == "color") throw new Exception("color param not found: " + name);
            }
            if (type == "" || type == "chooser")
            {
                JSONStorableStringChooser jch = st.GetStringChooserJSONParam(name);
                if (jch != null)
                {
                    jch.val = value.Value;
                    d["type"] = "chooser";
                    d["value"] = jch.val;
                    return d;
                }
                if (type == "chooser") throw new Exception("chooser param not found: " + name);
            }
            if (type == "" || type == "string")
            {
                JSONStorableString js = st.GetStringJSONParam(name);
                if (js != null)
                {
                    js.val = value.Value;
                    d["type"] = "string";
                    return d;
                }
                if (type == "string") throw new Exception("string param not found: " + name);
            }
            throw new Exception("param not found (any type): " + name);
        }

        private JSONClass CmdCallAction(JSONClass args)
        {
            JSONStorable st = RequireStorable(args["atom"].Value, args["storable"].Value);
            string name = args["action"].Value;
            st.CallAction(name); // TODO(verify) throws if missing? wrap either way
            JSONClass d = new JSONClass();
            d["called"] = name;
            return d;
        }

        // ------------------------------------------------------------------
        // Character (skin) selection
        // ------------------------------------------------------------------

        private DAZCharacterSelector RequireSelector(string atomUid)
        {
            Atom atom = RequireAtom(atomUid);
            DAZCharacterSelector selector = atom.GetComponentInChildren<DAZCharacterSelector>();
            if (selector == null) throw new Exception("atom is not a Person: " + atomUid);
            return selector;
        }

        private JSONClass CmdListCharacters(JSONClass args)
        {
            DAZCharacterSelector selector = RequireSelector(args["atom"].Value);
            JSONArray arr = new JSONArray();
            DAZCharacter[] characters = selector.characters; // TODO(verify) member name
            if (characters != null)
            {
                for (int i = 0; i < characters.Length; i++)
                {
                    if (characters[i] != null) arr.Add(new JSONData(characters[i].displayName));
                }
            }
            JSONClass d = new JSONClass();
            d["characters"] = arr;
            return d;
        }

        private JSONClass CmdSetCharacter(JSONClass args)
        {
            DAZCharacterSelector selector = RequireSelector(args["atom"].Value);
            string name = args["name"].Value;
            selector.SelectCharacterByName(name); // TODO(verify) method name
            JSONClass d = new JSONClass();
            d["selected"] = name;
            return d;
        }

        // ------------------------------------------------------------------
        // Screenshot (async, replies from coroutine)
        //
        // v0.5.5: primary path renders the scene camera into an offscreen
        // RenderTexture (no screen-space UI, no toolbar, exact requested
        // width, 4x MSAA). The old whole-screen ReadPixels path is kept as
        // an automatic fallback — a fit run must never die because camera
        // discovery failed on some VaM setup.
        // ------------------------------------------------------------------

        // Find the camera rendering the desktop view. Runtime probing only —
        // deliberately no SuperController camera members here (their names
        // are unverified on 1.22 and a bad name kills compilation; see the
        // System.IO lesson). Fallbacks keep this null-safe.
        private Camera FindSceneCamera()
        {
            Camera cam = Camera.main;
            if (cam != null && cam.enabled) return cam;
            GameObject go = GameObject.Find("MonitorCenterCamera");
            if (go != null)
            {
                Camera c = go.GetComponent<Camera>();
                if (c != null && c.enabled) return c;
            }
            // Highest-depth enabled camera that renders to the screen.
            Camera best = null;
            Camera[] all = Camera.allCameras;
            for (int i = 0; i < all.Length; i++)
            {
                if (all[i].targetTexture != null) continue; // offscreen cam
                if (best == null || all[i].depth > best.depth) best = all[i];
            }
            return best;
        }

        private byte[] CaptureFromCamera(Camera cam, int maxWidth,
                                         out int outW, out int outH)
        {
            // Keep the on-screen aspect so framing matches what focus_head
            // set up; render directly at the requested width (crisper than
            // downscaling a screen grab, and UI-free by construction).
            int w = maxWidth > 0 ? maxWidth : Screen.width;
            int h = Mathf.RoundToInt((float)Screen.height * w / Screen.width);
            RenderTexture rt = RenderTexture.GetTemporary(
                w, h, 24, RenderTextureFormat.Default,
                RenderTextureReadWrite.Default, 4);
            RenderTexture prevTarget = cam.targetTexture;
            RenderTexture prevActive = RenderTexture.active;
            int prevMask = cam.cullingMask;
            try
            {
                cam.targetTexture = rt;
                cam.cullingMask = prevMask & ~LayerMask.GetMask("UI");
                cam.Render();
                RenderTexture.active = rt;
                Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
                tex.Apply();
                byte[] png = tex.EncodeToPNG();
                UnityEngine.Object.Destroy(tex);
                outW = w; outH = h;
                return png;
            }
            finally
            {
                cam.targetTexture = prevTarget;
                cam.cullingMask = prevMask;
                RenderTexture.active = prevActive;
                RenderTexture.ReleaseTemporary(rt);
            }
        }

        private IEnumerator CaptureRoutine(ClientConn conn, string id, JSONClass args)
        {
            int maxWidth = args["max_width"] != null ? args["max_width"].AsInt : 0;
            // camera arg: default true; {"camera": false} forces the old
            // whole-screen grab (debug / comparison).
            bool wantCamera = args["camera"] == null || args["camera"].AsBool;

            yield return new WaitForEndOfFrame();

            byte[] png = null;
            string error = null;
            string method = "screen";
            int outW = 0, outH = 0;

            if (wantCamera)
            {
                try
                {
                    Camera cam = FindSceneCamera();
                    if (cam != null)
                    {
                        png = CaptureFromCamera(cam, maxWidth, out outW, out outH);
                        method = "camera";
                    }
                }
                catch (Exception e)
                {
                    SuperController.LogMessage(
                        "VamFaceBridge: camera capture failed, " +
                        "falling back to screen grab: " + e.Message);
                    png = null;
                }
            }

            if (png == null)
            {
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
                    method = "screen";
                }
                catch (Exception e)
                {
                    error = "screenshot failed: " + e.Message;
                }
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
                d["method"] = method;
                d["png_base64"] = Convert.ToBase64String(png);
                Reply(conn, id, d);
            }
        }
    }
}
