"""可切换的打分器栈 —— 解决"anime 脸 ArcFace 失效"的根本问题。

背景(见 对话记忆):ArcFace 在真人照片上训练,对二次元/风格化脸的 embedding
基本是噪声,拟合循环会朝无意义的方向优化。所以打分器必须按素材风格可切换:

  style="real"   ArcFace 身份相似度(主)+ 五官几何损失(辅,若有 landmark)
  style="anime"  anime 五官几何损失(animeface 检测器)+ 可选的用户自备
                 anime 识别 ONNX 模型 embedding
  style="pixel"  纯像素相似度(降采样灰度 MSE)。跨域(照片 vs 3D 渲染)时
                 意义有限,但对 mock server 的端到端测试、以及"同域"比较
                 (渲染 vs 渲染)是完全够用的粗打分器。
  style="auto"   用两个检测器探测目标图:anime 检出→anime,真人检出→real,
                 都不行→pixel(带 warning,别默默装有效)。

几何打分器额外提供**方向性提示**(GeometryScorer.hints()):
"目标两眼间距更大 → Eyes Width Spacing ↑"这类可读建议。纯黑盒分数只会说
"不像",几何特征差能说"哪里不像、往哪调",这既能给人看,以后也能喂给
带先验的优化器。

降级铁律(教训5):任何依赖装不上都不 crash —— 降一级并把 warning 带出去,
让调用方知道分数的含金量。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("vamface.scorers")

FeatureDict = Dict[str, float]


# ---------------------------------------------------------------------------
# 基类与两个零依赖打分器
# ---------------------------------------------------------------------------

class Scorer:
    """约定:score() 返回越大越相似,理想范围 [0, 1]。"""

    name = "base"

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        raise NotImplementedError

    def hints(self) -> List[str]:
        """最近一次评估的方向性提示(没有就空列表)。"""
        return []


class NullScorer(Scorer):
    """占位:恒 0。让整条流水线在没装任何依赖时也能干跑。"""

    name = "null"
    reason = "no scorer available for this style"

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        return 0.0


class PixelScorer(Scorer):
    """降采样灰度负 MSE → [0,1] 相似度。

    零依赖、光滑、快。适用:mock 端到端测试、渲染-对-渲染的同域比较、
    以及所有检测器都失效时的最后兜底(带 warning)。
    跨域(真实照片 vs 3D 渲染)时它更多在比构图/影调,别指望它比五官。
    """

    name = "pixel"

    def __init__(self, size: int = 64) -> None:
        self.size = int(size)
        self._target_cache: Optional[Tuple[int, np.ndarray]] = None

    def _prep(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(img).convert("L").resize(
            (self.size, self.size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return arr

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        key = id(target)
        if self._target_cache is None or self._target_cache[0] != key:
            self._target_cache = (key, self._prep(target))
        t = self._target_cache[1]
        c = self._prep(candidate)
        rmse = float(np.sqrt(np.mean((t - c) ** 2)))
        # 用 RMSE 而非 MSE:MSE 在"已经比较像"的区间里会把分数压扁在 1 附近,
        # 优化末段几乎没有区分度;RMSE 标度下 base≈0.75、good fit≈0.95,
        # 动态范围合理且依旧光滑。
        return float(np.exp(-10.0 * rmse))


# ---------------------------------------------------------------------------
# 五官几何特征:从 landmark 提取归一化特征向量
# ---------------------------------------------------------------------------
#
# 特征全部用人脸框归一化,和分辨率/位置无关:
#   eye_gap     两眼中心水平距 / 脸宽        ← Eyes Width Spacing
#   eye_w       平均眼宽 / 脸宽              ← Eyes Size
#   eye_h       平均眼高 / 脸高              ← Eyes Size / Eyelids Height
#   eye_y       眼中心在脸框内的相对高度      ← Eyes Height / Brow Height
#   nose_len    眼线到鼻尖距离 / 脸高         ← Nose Size / Nose Height
#   mouth_w     嘴宽 / 脸宽                  ← Mouth Width / Lips Width
#   mouth_y     嘴中心相对高度               ← Mouth Height
#   jaw_len     嘴到下巴距离 / 脸高(有 chin 时)← Chin Height / Jaw Size
#   face_aspect 脸高 / 脸宽                  ← Face Long / Face Round
#
# 提取器签名统一:extractor(img) -> Optional[FeatureDict]
# 检出失败返回 None(不是异常)。不同后端能给的键不一样,比较时只比交集。

# 特征差 → 提示的中文名与方向附注。
# v0.5.4 起:**morph 名和箭头方向不再写死在字符串里**,渲染时从
# priors.FEATURE_TO_MORPHS 的增益符号推导(Δ>0 且 gain>0 → ↑,以此类推)。
# 方向只有一个真相源,"两张表同步改"的 grep 规矩作废。
# 这也让提示能按目标 VaM 的实际可用性过滤/改名(见
# GeometryScorer.set_morph_availability)—— 真机第三跑抓到的 bug:
# 提示推荐了 missing 列表里的 morph,推荐一根不存在的滑块。
_HINT_MAP: Dict[str, Tuple[str, str, str]] = {
    # feature: (中文名, Δ>0 附注, Δ<0 附注)
    "eye_gap": ("两眼间距", "", ""),
    "eye_w": ("眼睛大小(宽)", "", ""),
    "eye_h": ("眼睛开度(高)", "", ""),
    "eye_y": ("眼睛位置", "(眼更靠下)", "(眼更靠上)"),
    "nose_len": ("鼻长", "", ""),
    "mouth_w": ("嘴宽", "", ""),
    "mouth_y": ("嘴位置", "(嘴更靠下)", "(嘴更靠上)"),
    "jaw_len": ("下巴长度", "", ""),
    "face_aspect": ("脸型长宽比", "", ""),
}


def features_from_animeface(img: np.ndarray) -> Optional[FeatureDict]:
    """animeface(PyPI 包,nyanp/animeface-2009 绑定)→ 特征向量。

    animeface.detect 返回的 face 对象带 face/left_eye/right_eye/nose/mouth/chin
    的 pos。字段按防御式 getattr 取 —— 该库年头久,别硬信字段一定在。
    """
    import animeface  # 懒加载
    from PIL import Image

    faces = animeface.detect(Image.fromarray(img))
    if not faces:
        return None
    f = max(faces, key=lambda x: getattr(x, "likelihood", 0.0))

    def box(part):  # -> (cx, cy, w, h) or None
        pos = getattr(part, "pos", None) if part is not None else None
        if pos is None:
            return None
        w = float(getattr(pos, "width", 0.0))
        h = float(getattr(pos, "height", 0.0))
        return (float(pos.x) + w / 2.0, float(pos.y) + h / 2.0, w, h)

    face = box(getattr(f, "face", None))
    le, re = box(getattr(f, "left_eye", None)), box(getattr(f, "right_eye", None))
    nose = box(getattr(f, "nose", None))     # nose 只有点,w/h 为 0
    mouth = box(getattr(f, "mouth", None))
    chin = box(getattr(f, "chin", None))
    if face is None or le is None or re is None:
        return None

    fx, fy, fw, fh = face
    fw = max(fw, 1.0)
    fh = max(fh, 1.0)
    eye_cy = (le[1] + re[1]) / 2.0
    feat: FeatureDict = {
        "eye_gap": abs(re[0] - le[0]) / fw,
        "eye_w": (le[2] + re[2]) / 2.0 / fw,
        "eye_h": (le[3] + re[3]) / 2.0 / fh,
        "eye_y": (eye_cy - (fy - fh / 2.0)) / fh,
        "face_aspect": fh / fw,
    }
    if nose is not None:
        feat["nose_len"] = (nose[1] - eye_cy) / fh
    if mouth is not None:
        feat["mouth_w"] = mouth[2] / fw
        feat["mouth_y"] = (mouth[1] - (fy - fh / 2.0)) / fh
        if chin is not None:
            feat["jaw_len"] = (chin[1] - mouth[1]) / fh
    return feat


class _InsightFeatureExtractor:
    """insightface → 特征向量(真人脸)。优先 106 点 landmark,退化用 5 点 kps。"""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis  # 懒加载、重依赖

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, img: np.ndarray) -> Optional[FeatureDict]:
        faces = self.app.get(img[:, :, ::-1])  # insightface 吃 BGR
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        fw, fh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        kps = np.asarray(f.kps, dtype=float)  # 5 点: 左眼 右眼 鼻尖 左嘴角 右嘴角
        le, re, nose, ml, mr = kps
        eye_cy = (le[1] + re[1]) / 2.0
        feat: FeatureDict = {
            "eye_gap": abs(re[0] - le[0]) / fw,
            "eye_y": (eye_cy - y1) / fh,
            "nose_len": (nose[1] - eye_cy) / fh,
            "mouth_w": abs(mr[0] - ml[0]) / fw,
            "mouth_y": ((ml[1] + mr[1]) / 2.0 - y1) / fh,
            "face_aspect": fh / fw,
        }
        lmk = getattr(f, "landmark_2d_106", None)
        if lmk is not None:
            lmk = np.asarray(lmk, dtype=float)
            # 106 点里眼睛轮廓:左眼 33-42、右眼 87-96(insightface 惯例)。
            # TODO(verify-lib): 不同版本索引可能变,取不到就跳过这两个特征。
            try:
                for key, sl in (("_l", slice(33, 43)), ("_r", slice(87, 97))):
                    pts = lmk[sl]
                    feat["eye_w" + key] = float(np.ptp(pts[:, 0])) / fw
                    feat["eye_h" + key] = float(np.ptp(pts[:, 1])) / fh
                feat["eye_w"] = (feat.pop("eye_w_l") + feat.pop("eye_w_r")) / 2.0
                feat["eye_h"] = (feat.pop("eye_h_l") + feat.pop("eye_h_r")) / 2.0
            except Exception:
                pass
        return feat


class GeometryScorer(Scorer):
    """五官几何相似度:1 提取双方特征 2 只比交集键 3 exp(-k·加权平均|Δ|)。

    extractor 可注入 —— 测试时喂合成特征函数,生产时喂 animeface/insightface。
    """

    name = "geometry"

    # 各特征的权重(量纲已归一,权重只表达"该维度对相似的重要性")
    WEIGHTS: Dict[str, float] = {
        "eye_gap": 2.0, "eye_w": 2.0, "eye_h": 1.5, "eye_y": 1.0,
        "nose_len": 1.0, "mouth_w": 1.5, "mouth_y": 1.0,
        "jaw_len": 1.0, "face_aspect": 2.0,
    }

    def __init__(self, extractor: Callable[[np.ndarray], Optional[FeatureDict]],
                 sharpness: float = 8.0) -> None:
        self.extractor = extractor
        self.sharpness = float(sharpness)
        self._target_cache: Optional[Tuple[int, Optional[FeatureDict]]] = None
        self.last_diff: FeatureDict = {}   # 最近一次 candidate 相对 target 的差
        self.detect_misses = 0             # candidate 检不出脸的累计次数
        # 目标 VaM 的 morph 可用性(None = 未知,不过滤)与 概念名→实际名 改名表
        self._avail_norm: Optional[set] = None
        self._hint_rename: Dict[str, str] = {}

    def set_morph_availability(self, available=None, rename=None) -> None:
        """告知目标 VaM 实际有哪些 morph(hints 据此过滤/改名)。

        available: 实际 morph 名的可迭代;None = 恢复"不过滤"
        rename: 概念名 → 实际名(别名解析的产物),提示里显示实际名
        """
        from .morph_presets import norm_name
        self._avail_norm = ({norm_name(n) for n in available}
                            if available is not None else None)
        self._hint_rename = dict(rename or {})

    def _extract(self, img: np.ndarray) -> Optional[FeatureDict]:
        # 依赖缺失在构建期就被拦下了;这里的异常是单张图上的检测器抖动,
        # 按"没检出"处理(计数 + 0 分),不许杀掉一次长拟合(教训5)。
        try:
            return self.extractor(img)
        except Exception:
            log.warning("landmark extractor failed on one image", exc_info=True)
            return None

    def _target_features(self, target: np.ndarray) -> Optional[FeatureDict]:
        key = id(target)
        if self._target_cache is None or self._target_cache[0] != key:
            self._target_cache = (key, self._extract(target))
        return self._target_cache[1]

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        tf = self._target_features(target)
        if tf is None:
            return 0.0
        cf = self._extract(candidate)
        if cf is None:
            self.detect_misses += 1
            return 0.0
        keys = sorted(set(tf) & set(cf) & set(self.WEIGHTS))
        if not keys:
            return 0.0
        self.last_diff = {k: tf[k] - cf[k] for k in keys}  # 正 = 目标更大
        num = sum(self.WEIGHTS[k] * abs(self.last_diff[k]) for k in keys)
        den = sum(self.WEIGHTS[k] for k in keys)
        return float(np.exp(-self.sharpness * num / den))

    def hints(self, threshold: float = 0.02) -> List[str]:
        from .morph_presets import norm_name
        from .priors import FEATURE_TO_MORPHS
        out: List[str] = []
        for k, d in sorted(self.last_diff.items(), key=lambda kv: -abs(kv[1])):
            if abs(d) < threshold or k not in _HINT_MAP:
                continue
            zh, note_pos, note_neg = _HINT_MAP[k]
            note = note_pos if d > 0 else note_neg
            moves: List[str] = []
            dropped: List[str] = []
            for name, gain in FEATURE_TO_MORPHS.get(k, []):
                shown = self._hint_rename.get(name, name)
                if (self._avail_norm is not None
                        and norm_name(shown) not in self._avail_norm):
                    dropped.append(shown)
                    continue
                up = (gain > 0) == (d > 0)
                moves.append(f"{shown} {'↑' if up else '↓'}")
            if moves:
                out.append(f"{zh}差 {d:+.3f} → {' / '.join(moves)}{note}")
            elif dropped:
                out.append(f"{zh}差 {d:+.3f} → 对应 morph 你的 VaM 缺失"
                           f"({' / '.join(dropped)}),没法自动修")
            else:
                out.append(f"{zh}差 {d:+.3f}{note}")
        return out


# ---------------------------------------------------------------------------
# embedding 打分器:ArcFace(真人)与用户自备 ONNX(anime)
# ---------------------------------------------------------------------------

class ArcFaceScorer(Scorer):
    """ArcFace embedding 余弦相似度(真人脸的定量主路径)。"""

    name = "arcface"

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis  # 懒加载、重依赖

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._target_cache: Optional[np.ndarray] = None

    def _embed(self, img: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(img[:, :, ::-1])
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                   reverse=True)
        return np.asarray(faces[0].normed_embedding, dtype=np.float32)

    def set_target(self, target: np.ndarray) -> bool:
        self._target_cache = self._embed(target)
        return self._target_cache is not None

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        if self._target_cache is None and not self.set_target(target):
            return 0.0
        cand = self._embed(candidate)
        if cand is None:
            return 0.0
        return float(np.dot(self._target_cache, cand))


class OnnxEmbeddingScorer(Scorer):
    """用户自备的人脸(通常是 anime 识别)ONNX 模型 → embedding 余弦相似度。

    约定:模型输入 NCHW float32、RGB、[0,1] 或 [-1,1](normalize 参数),
    输出一条 embedding 向量。社区训练的动漫识别模型基本都长这样;
    不满足约定就报 warning 降级,不 crash。
    """

    name = "onnx-embed"

    def __init__(self, model_path: str, input_size: int = 224,
                 normalize: str = "0-1") -> None:
        import onnxruntime as ort  # 懒加载

        self.sess = ort.InferenceSession(model_path,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = int(input_size)
        self.normalize = normalize
        self._target_cache: Optional[np.ndarray] = None

    def _embed(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(img).resize((self.input_size, self.input_size),
                                          Image.BILINEAR)
        x = np.asarray(pil, dtype=np.float32) / 255.0
        if self.normalize == "-1-1":
            x = x * 2.0 - 1.0
        x = np.transpose(x, (2, 0, 1))[None]  # NCHW
        emb = self.sess.run(None, {self.input_name: x})[0].reshape(-1)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        if self._target_cache is None:
            self._target_cache = self._embed(target)
        return float(np.dot(self._target_cache, self._embed(candidate)))


# ---------------------------------------------------------------------------
# 组合与构建
# ---------------------------------------------------------------------------

class CompositeScorer(Scorer):
    """加权平均若干子打分器;hints 汇总子打分器的方向性提示。"""

    name = "composite"

    def __init__(self, parts: Sequence[Tuple[Scorer, float]]) -> None:
        total = sum(w for _, w in parts) or 1.0
        self.parts: List[Tuple[Scorer, float]] = [(s, w / total) for s, w in parts]
        self.name = "+".join(f"{s.name}×{w:.2f}" for s, w in self.parts)

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        return float(sum(w * s.score(target, candidate) for s, w in self.parts))

    def hints(self) -> List[str]:
        out: List[str] = []
        for s, _ in self.parts:
            out.extend(s.hints())
        return out


@dataclass
class ScorerBuild:
    scorer: Scorer
    style: str                      # 实际生效的风格(auto 解析后的结果)
    warnings: List[str] = field(default_factory=list)

    @property
    def warning(self) -> Optional[str]:
        return "; ".join(self.warnings) or None


def _try(factory: Callable[[], Scorer], label: str,
         warnings: List[str]) -> Optional[Scorer]:
    try:
        return factory()
    except Exception as e:  # ImportError / 模型下载失败 / onnx 不合约定
        log.warning("%s 不可用: %s", label, e)
        warnings.append(f"{label} 不可用: {e}")
        return None


def build_scorer_stack(style: str = "auto",
                       target: Optional[np.ndarray] = None,
                       anime_onnx: Optional[str] = None) -> ScorerBuild:
    """按素材风格构建打分器,永不抛异常。

    style: auto | real | anime | pixel
    target: auto 模式下用来探测风格的目标图(不给就无法 auto,落到 pixel)
    anime_onnx: 可选的 anime 识别 ONNX 模型路径(anime 模式的 embedding 部分)
    """
    warnings: List[str] = []

    if style == "auto":
        style = _detect_style(target, warnings)

    if style == "pixel":
        # PixelScorer 吃整图,最怕背景污染 —— 包一层零依赖的主体框裁剪
        return ScorerBuild(CroppedScorer(PixelScorer(), bbox_from_background),
                           "pixel", warnings)

    if style == "anime":
        parts: List[Tuple[Scorer, float]] = []
        def _make_anime_geo() -> Scorer:
            import animeface  # noqa: F401 — 构建期即验证依赖,别把 ImportError 拖到打分时
            return GeometryScorer(features_from_animeface)

        geo = _try(_make_anime_geo,
                   "animeface 几何打分器(pip install animeface)", warnings)
        if geo is not None:
            parts.append((geo, 1.0))
        if anime_onnx:
            emb = _try(lambda: OnnxEmbeddingScorer(anime_onnx),
                       f"anime ONNX embedding({anime_onnx})", warnings)
            if emb is not None:
                # ONNX embedding 也是整图 resize 进模型 —— 用 anime 人脸框裁剪
                parts.append((CroppedScorer(emb, box_from_animeface), 1.0))
        if parts:
            scorer = parts[0][0] if len(parts) == 1 else CompositeScorer(parts)
            return ScorerBuild(scorer, "anime", warnings)
        warnings.append("anime 打分全灭,降级 pixel(分数只反映构图/影调,慎信)")
        return ScorerBuild(CroppedScorer(PixelScorer(), bbox_from_background),
                           "pixel", warnings)

    if style == "real":
        arc = _try(ArcFaceScorer, "ArcFace(pip install -e '.[fit]')", warnings)
        geo: Optional[Scorer] = None
        if arc is not None:
            # 复用同一个 FaceAnalysis,省一次模型加载
            ext = _InsightFeatureExtractor.__new__(_InsightFeatureExtractor)
            ext.app = arc.app
            geo = GeometryScorer(ext)
            return ScorerBuild(CompositeScorer([(arc, 0.75), (geo, 0.25)]),
                               "real", warnings)
        geo = _try(lambda: GeometryScorer(_InsightFeatureExtractor()),
                   "insightface 几何打分器", warnings)
        if geo is not None:
            warnings.append("ArcFace 不可用,仅几何打分(区分度下降)")
            return ScorerBuild(geo, "real", warnings)
        warnings.append("real 打分全灭,降级 NullScorer(分数恒 0,结果无意义)")
        return ScorerBuild(NullScorer(), "null", warnings)

    warnings.append(f"未知 style '{style}',降级 NullScorer")
    return ScorerBuild(NullScorer(), "null", warnings)


def _detect_style(target: Optional[np.ndarray], warnings: List[str]) -> str:
    """auto:先 anime 检测器,再真人检测器,都不行 pixel。"""
    if target is None:
        warnings.append("auto 模式没有目标图可探测,落到 pixel")
        return "pixel"
    try:
        if features_from_animeface(target) is not None:
            return "anime"
    except Exception as e:
        warnings.append(f"anime 探测不可用: {e}")
    try:
        ext = _InsightFeatureExtractor()
        if ext(target) is not None:
            return "real"
    except Exception as e:
        warnings.append(f"真人探测不可用: {e}")
    warnings.append("两类检测器都没在目标图上检出脸,落到 pixel")
    return "pixel"


# ---------------------------------------------------------------------------
# 人脸对齐裁剪(v0.4)—— 别让背景/构图污染分数
# ---------------------------------------------------------------------------
#
# 谁需要裁剪要想清楚,不是无脑全包:
#   - ArcFaceScorer / GeometryScorer 内部本来就先做人脸检测(insightface 还
#     自带 landmark 对齐),再包一层裁剪是白花一次检测;
#   - **PixelScorer 和 OnnxEmbeddingScorer 是真正的受害者**:它们吃整张图,
#     换个背景/灯光/构图分数就飘。给它们包 CroppedScorer。
# 降级铁律不变:检测不到框就退回整图 + 计数,不 crash、不清零。

def bbox_from_background(img: np.ndarray, tol: int = 24,
                         margin: float = 0.12) -> Optional[Tuple[int, int, int, int]]:
    """零依赖找主体框:与边框底色差超过 tol 的像素的包围盒。

    对纯色背景的渲染图(mock、"证件照模式"拟合场景)非常好使;
    对真实照片基本没用 —— 那是检测器的活,这个函数别越界。
    返回 (x0, y0, x1, y1) 或 None(找不到主体)。
    """
    a = img.astype(np.int16)
    border = np.concatenate([a[0].reshape(-1, 3), a[-1].reshape(-1, 3),
                             a[:, 0].reshape(-1, 3), a[:, -1].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    mask = np.abs(a - bg).sum(axis=2) > tol
    if mask.sum() < 25:
        return None
    ys, xs = np.nonzero(mask)
    h, w = img.shape[:2]
    mx, my = int(margin * (xs.max() - xs.min())), int(margin * (ys.max() - ys.min()))
    return (max(0, xs.min() - mx), max(0, ys.min() - my),
            min(w, xs.max() + mx + 1), min(h, ys.max() + my + 1))


def box_from_animeface(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """animeface 人脸框(含 margin)。"""
    import animeface
    from PIL import Image

    faces = animeface.detect(Image.fromarray(img))
    if not faces:
        return None
    f = max(faces, key=lambda x: getattr(x, "likelihood", 0.0))
    pos = getattr(getattr(f, "face", None), "pos", None)
    if pos is None:
        return None
    h, w = img.shape[:2]
    mx, my = int(0.25 * pos.width), int(0.25 * pos.height)
    return (max(0, int(pos.x) - mx), max(0, int(pos.y) - my),
            min(w, int(pos.x + pos.width) + mx),
            min(h, int(pos.y + pos.height) + my))


class CroppedScorer(Scorer):
    """打分前把 target/candidate 都裁到主体框,再交给内层打分器。

    box_fn(img) -> (x0,y0,x1,y1) | None。None 或异常 → 用整图(降级 + 计数)。
    target 的框按 id 缓存,一次拟合只检测一次。
    """

    def __init__(self, inner: Scorer,
                 box_fn: Callable[[np.ndarray], Optional[Tuple[int, int, int, int]]]) -> None:
        self.inner = inner
        self.box_fn = box_fn
        self.name = f"crop({inner.name})"
        self.crop_misses = 0
        self._target_cache: Optional[Tuple[int, np.ndarray]] = None

    def _crop(self, img: np.ndarray) -> np.ndarray:
        try:
            box = self.box_fn(img)
        except Exception:
            log.warning("crop box_fn failed on one image", exc_info=True)
            box = None
        if box is None:
            self.crop_misses += 1
            return img
        x0, y0, x1, y1 = box
        return img[y0:y1, x0:x1]

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        key = id(target)
        if self._target_cache is None or self._target_cache[0] != key:
            self._target_cache = (key, self._crop(target))
        return self.inner.score(self._target_cache[1], self._crop(candidate))

    def hints(self) -> List[str]:
        return self.inner.hints()


def find_geometry_scorer(scorer: Scorer) -> Optional[GeometryScorer]:
    """从(可能嵌套的)打分器里挖出 GeometryScorer —— 先验要用它的 last_diff。"""
    if isinstance(scorer, GeometryScorer):
        return scorer
    if isinstance(scorer, CroppedScorer):
        return find_geometry_scorer(scorer.inner)
    if isinstance(scorer, CompositeScorer):
        for s, _ in scorer.parts:
            g = find_geometry_scorer(s)
            if g is not None:
                return g
    return None
