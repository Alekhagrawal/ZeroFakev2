# File: discord_bot.py
# What this file does: Connects a Discord bot to ZeroFake so Discord users can verify face photos
#   or short video clips directly inside any Discord server channel using:
#     1. The slash command `/check` with an image or video `discord.Attachment` (immediately calling
#        `await interaction.response.defer(thinking=True)` so Discord never times out during inference).
#     2. Mentioning `@ZeroFake` in any message (or replying `@ZeroFake` to an earlier message) that has
#        an image or video attachment.
#   Supports two inference backends automatically:
#     - Remote Gradio API Mode: When `ZEROFAKE_URL` is set (e.g., `https://*.gradio.live` or an AWS App Runner URL),
#       calls `gradio_client.Client(ZEROFAKE_URL)` with `handle_file(...)` targeting `api_name="/predict_media"`.
#     - Local In-Process Mode: When `ZEROFAKE_URL` is not set and `checkpoints/zerofake/best_model.pt` exists
#       locally (or in local preview mode), runs `ZeroFakeDemoEngine.predict_media_api(...)` directly in-process.
# Inputs:
#   - Environment variables (loaded automatically from `.env` via `python-dotenv` if present):
#       * `DISCORD_BOT_TOKEN`: Discord bot authentication token.
#       * `ZEROFAKE_URL`: Optional live ZeroFake Gradio URL (`https://*.gradio.live` or `http://127.0.0.1:7860`).
#   - `--dry-run` (CLI flag): Verifies embed construction, color coding (`0x16a34a` REAL, `0xdc2626` FAKE,
#     `0xd97706` UNKNOWN), `attachment://heatmap.png` wiring, and in-process `predict_media_api` inference
#     on CPU in under 5 seconds (Rule 5).
# Outputs:
#   - Color-coded Discord Embed showing the Verdict badge, Real vs. Fake lean percentages, Face Detection
#     status, plain-English summary sentence, and the attached Grad-CAM Heatmap PNG (`attachment://heatmap.png`).

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import asyncio
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    load_dotenv = None
    HAS_DOTENV = False

try:
    import discord
    from discord import app_commands
    HAS_DISCORD = True
except ImportError:
    discord = None
    app_commands = None
    HAS_DISCORD = False

try:
    from gradio_client import Client, handle_file
    HAS_GRADIO_CLIENT = True
except ImportError:
    Client = None
    handle_file = None
    HAS_GRADIO_CLIENT = False


# Exact color palette matching app.py's 3-way Executive Verdict Card
COLOR_REAL_GREEN = 0x16A34A    # Emerald green (#16a34a) for REAL
COLOR_FAKE_RED = 0xDC2626      # Crimson red (#dc2626) for FAKE
COLOR_UNKNOWN_AMBER = 0xD97706 # Amber (#d97706) for UNKNOWN

SUPPORTED_MEDIA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
}


def get_verdict_style(verdict: str) -> Tuple[int, str, str]:
    """
    Maps a ZeroFake 3-way verdict ('REAL', 'FAKE', or 'UNKNOWN') to:
      1. Discord embed hex color integer (`0x16a34a`, `0xdc2626`, or `0xd97706`)
      2. Verdict badge title
      3. Subtitle description
    """
    v = (verdict or "UNKNOWN").strip().upper()
    if v == "REAL":
        return COLOR_REAL_GREEN, "✅ REAL — Authentic Face", "High-Confidence Authentic Face"
    if v == "FAKE":
        return COLOR_FAKE_RED, "🚨 FAKE — Deepfake Detected", "High-Confidence Synthetic / Manipulated Face"
    return COLOR_UNKNOWN_AMBER, "⚠️ UNKNOWN — Unclear or Unfamiliar Face", "Open-Set Rejection (Flagged for Human Review)"


def format_progress_bar(pct: float, width: int = 12) -> str:
    """
    Creates a compact Unicode horizontal comparison bar for Discord embed fields.
    """
    clamped = max(0.0, min(100.0, float(pct)))
    filled = int(round((clamped / 100.0) * width))
    empty = max(0, width - filled)
    return "█" * filled + "░" * empty


def build_verdict_embed(
    result: Dict[str, Any],
    filename: str = "media",
    has_heatmap: bool = False,
) -> Any:
    """
    Constructs a clean, color-coded Discord Embed from ZeroFake's `predict_media` dictionary:
      - Emerald green (`0x16a34a`) for `REAL`, Crimson red (`0xdc2626`) for `FAKE`, Amber (`0xd97706`) for `UNKNOWN`.
      - Shows the Verdict badge, Real vs. Fake lean percentages, Face Detection status, plain-English summary,
        and embeds `attachment://heatmap.png` when a Grad-CAM heatmap PNG is attached.
    """
    raw_verdict = str(result.get("verdict", "UNKNOWN")).strip().upper()
    verdict = raw_verdict if raw_verdict in ("REAL", "FAKE", "UNKNOWN") else "UNKNOWN"
    real_pct = float(result.get("real_pct", 0.0))
    fake_pct = float(result.get("fake_pct", 0.0))
    summary = str(result.get("summary", "Open-set deepfake analysis complete."))
    face_status = str(result.get("face_status", "Analyzed by ZeroFake MTCNN + ConvNeXt-Tiny"))

    color_hex, title_text, subtitle_text = get_verdict_style(verdict)

    if not HAS_DISCORD or discord is None:
        return {
            "title": title_text,
            "subtitle": subtitle_text,
            "color": color_hex,
            "verdict": verdict,
            "real_pct": real_pct,
            "fake_pct": fake_pct,
            "summary": summary,
            "face_status": face_status,
            "filename": filename,
            "image_url": "attachment://heatmap.png" if has_heatmap else "",
        }

    embed = discord.Embed(
        title=title_text,
        description=f"**{subtitle_text}**\n{summary}",
        color=discord.Color(color_hex),
    )

    real_bar = format_progress_bar(real_pct)
    fake_bar = format_progress_bar(fake_pct)
    embed.add_field(
        name="✓ Authentic (Real) Lean",
        value=f"`{real_bar}` **{real_pct:.1f}%**",
        inline=True,
    )
    embed.add_field(
        name="✕ AI-Generated (Fake) Lean",
        value=f"`{fake_bar}` **{fake_pct:.1f}%**",
        inline=True,
    )
    embed.add_field(
        name="Scan Insights",
        value=(
            f"**Verdict Badge:** `{verdict}`\n"
            f"**Face Detection:** {face_status}\n"
            f"**Source File:** `{filename}`"
        ),
        inline=False,
    )

    if has_heatmap:
        embed.set_image(url="attachment://heatmap.png")

    embed.set_footer(
        text="ZeroFake • ConvNeXt-Tiny + 1 Real & 6 Fake Prototypes • Calibrated Open-Set Detection"
    )
    return embed


class ZeroFakePredictorBridge:
    """
    Unified prediction bridge for `discord_bot.py`:
      - If `zerofake_url` is set (e.g., `https://*.gradio.live` or AWS App Runner link), calls the remote
        ZeroFake API using `gradio_client.Client(zerofake_url)` and `handle_file(...)`.
      - If `zerofake_url` is not set and `checkpoints/zerofake/best_model.pt` exists locally (or fallback
        in-process mode is active), runs inference directly in-process via `app.ZeroFakeDemoEngine`.
    """

    def __init__(
        self,
        zerofake_url: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.zerofake_url = (zerofake_url or "").strip()
        self.default_local_ckpt = checkpoint_path or str(PROJECT_ROOT / "checkpoints" / "zerofake" / "best_model.pt")
        self._local_engine: Any = None

    @property
    def mode_description(self) -> str:
        if self.zerofake_url:
            return f"Remote Gradio API ({self.zerofake_url})"
        if os.path.exists(self.default_local_ckpt):
            return f"In-Process Local Checkpoint ({self.default_local_ckpt})"
        return "In-Process Local Engine (Preview Weights)"

    def _get_local_engine(self) -> Any:
        if self._local_engine is None:
            from app import ZeroFakeDemoEngine
            ckpt = self.default_local_ckpt if os.path.exists(self.default_local_ckpt) else None
            self._local_engine = ZeroFakeDemoEngine(checkpoint_path=ckpt)
        return self._local_engine

    def predict(self, local_media_path: str) -> Dict[str, Any]:
        """
        Runs deepfake detection on `local_media_path` either via remote `gradio_client` (when
        `ZEROFAKE_URL` is configured) or directly in-process.
        Returns a normalized dict with `verdict`, `real_pct`, `fake_pct`, `summary`, `face_status`,
        and `heatmap_path`.
        """
        if self.zerofake_url:
            return self._predict_via_gradio_client(local_media_path)
        return self._predict_in_process(local_media_path)

    def _predict_in_process(self, local_media_path: str) -> Dict[str, Any]:
        engine = self._get_local_engine()
        return engine.predict_media_api(local_media_path)

    def _predict_via_gradio_client(self, local_media_path: str) -> Dict[str, Any]:
        if not HAS_GRADIO_CLIENT or Client is None or handle_file is None:
            raise RuntimeError("gradio_client is not installed. Run `pip install gradio_client`.")

        client = Client(self.zerofake_url)
        raw_res = client.predict(
            handle_file(local_media_path),
            api_name="/predict_media",
        )

        # `api_name="predict_media"` returns either `(json_dict, downloaded_heatmap_filepath)` or `json_dict`
        payload_dict: Dict[str, Any] = {}
        downloaded_file: Optional[str] = None

        if isinstance(raw_res, (tuple, list)) and len(raw_res) >= 1:
            if isinstance(raw_res[0], dict):
                payload_dict = dict(raw_res[0])
            if len(raw_res) >= 2:
                second = raw_res[1]
                if isinstance(second, str):
                    downloaded_file = second
                elif isinstance(second, dict) and "path" in second:
                    downloaded_file = str(second["path"])
        elif isinstance(raw_res, dict):
            payload_dict = dict(raw_res)

        if not payload_dict:
            return {
                "verdict": "UNKNOWN",
                "real_pct": 50.0,
                "fake_pct": 50.0,
                "summary": f"Unexpected API response format from {self.zerofake_url}.",
                "face_status": "Unknown",
                "heatmap_path": "",
            }

        # Copy remote client's cached heatmap file into our own temp PNG so cleanup is safe
        heatmap_candidate = downloaded_file or str(payload_dict.get("heatmap_path", ""))
        safe_heatmap_path = ""
        if heatmap_candidate and os.path.exists(heatmap_candidate):
            fd, tmp_hm = tempfile.mkstemp(prefix="zf_discord_hm_", suffix=".png")
            os.close(fd)
            shutil.copy2(heatmap_candidate, tmp_hm)
            safe_heatmap_path = tmp_hm

        raw_verdict = str(payload_dict.get("verdict", "UNKNOWN")).strip().upper()
        verdict = raw_verdict if raw_verdict in ("REAL", "FAKE", "UNKNOWN") else "UNKNOWN"

        return {
            "verdict": verdict,
            "real_pct": float(payload_dict.get("real_pct", 0.0)),
            "fake_pct": float(payload_dict.get("fake_pct", 0.0)),
            "summary": str(payload_dict.get("summary", "")),
            "face_status": str(payload_dict.get("face_status", "")),
            "heatmap_path": safe_heatmap_path,
        }


def is_supported_attachment(filename: str, content_type: Optional[str] = None) -> bool:
    """
    Checks whether a Discord attachment is a supported image or video file.
    """
    ext = os.path.splitext((filename or "").lower())[1]
    if ext in SUPPORTED_MEDIA_EXTENSIONS:
        return True
    if content_type:
        ct = content_type.lower()
        if ct.startswith("image/") or ct.startswith("video/"):
            return True
    return False


if HAS_DISCORD and discord is not None:

    class ZeroFakeDiscordBot(discord.Client):
        """
        Discord bot supporting `/check <attachment>` slash commands and `@ZeroFake` message mentions.
        """

        def __init__(self, bridge: ZeroFakePredictorBridge) -> None:
            intents = discord.Intents.default()
            intents.message_content = True
            super().__init__(intents=intents)
            self.tree = app_commands.CommandTree(self)
            self.bridge = bridge

        async def setup_hook(self) -> None:
            await self.tree.sync()

        async def on_ready(self) -> None:
            print(f"[READY] Logged in as {self.user} (ID: {self.user.id})")
            print(f"[INFO] ZeroFake inference backend: {self.bridge.mode_description}")

        async def analyze_attachment_and_build_reply(
            self,
            attachment: discord.Attachment,
        ) -> Tuple[discord.Embed, Optional[discord.File], Optional[str]]:
            """
            Downloads a Discord attachment to a temporary file, runs ZeroFake inference in a worker
            thread so the Discord gateway heartbeat never blocks, and returns the color-coded
            `discord.Embed` plus `discord.File(..., filename="heatmap.png")` embedded as
            `attachment://heatmap.png`.
            """
            ext = os.path.splitext(attachment.filename or "upload.png")[1] or ".png"
            fd, tmp_input_path = tempfile.mkstemp(prefix="zf_discord_in_", suffix=ext)
            os.close(fd)

            try:
                await attachment.save(tmp_input_path)
                result = await asyncio.to_thread(self.bridge.predict, tmp_input_path)
            finally:
                if os.path.exists(tmp_input_path):
                    try:
                        os.remove(tmp_input_path)
                    except OSError:
                        pass

            heatmap_path = str(result.get("heatmap_path", ""))
            has_hm = bool(heatmap_path and os.path.exists(heatmap_path))
            embed = build_verdict_embed(result, filename=attachment.filename, has_heatmap=has_hm)

            discord_file: Optional[discord.File] = None
            if has_hm:
                discord_file = discord.File(heatmap_path, filename="heatmap.png")

            return embed, discord_file, (heatmap_path if has_hm else None)

        async def on_message(self, message: discord.Message) -> None:
            # Ignore messages from bots
            if message.author == self.user or message.author.bot:
                return

            # Trigger when @ZeroFake is mentioned in a message (or in a reply to a message with an attachment)
            if self.user is not None and self.user.mentioned_in(message):
                target_attachments = list(message.attachments)

                if not target_attachments and message.reference and message.reference.message_id:
                    try:
                        ref_msg = await message.channel.fetch_message(message.reference.message_id)
                        target_attachments = list(ref_msg.attachments)
                    except Exception:
                        target_attachments = []

                media_att = next(
                    (a for a in target_attachments if is_supported_attachment(a.filename, a.content_type)),
                    None,
                )
                if media_att is None:
                    await message.reply(
                        "Please attach a face image or short video clip (or reply `@ZeroFake` to a message "
                        "containing one), or use the `/check` slash command.",
                        mention_author=False,
                    )
                    return

                async with message.channel.typing():
                    try:
                        embed, d_file, hm_path = await self.analyze_attachment_and_build_reply(media_att)
                        try:
                            if d_file is not None:
                                await message.reply(embed=embed, file=d_file, mention_author=False)
                            else:
                                await message.reply(embed=embed, mention_author=False)
                        finally:
                            if hm_path and os.path.exists(hm_path):
                                try:
                                    os.remove(hm_path)
                                except OSError:
                                    pass
                    except Exception as exc:
                        await message.reply(
                            f"⚠️ **ZeroFake Inference Error:** `{exc}`",
                            mention_author=False,
                        )


def create_discord_bot(
    zerofake_url: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Any:
    """
    Creates the `ZeroFakeDiscordBot` client and registers the `/check` slash command.
    """
    if not HAS_DISCORD or discord is None:
        raise RuntimeError("discord.py (>=2.3) is not installed. Run `pip install discord.py`.")

    bridge = ZeroFakePredictorBridge(zerofake_url=zerofake_url, checkpoint_path=checkpoint_path)
    bot = ZeroFakeDiscordBot(bridge=bridge)

    @bot.tree.command(
        name="check",
        description="Scan a face image or short video clip with ZeroFake (REAL, FAKE, or UNKNOWN).",
    )
    @app_commands.describe(attachment="Image or video clip to scan for deepfake manipulation.")
    async def check_command(interaction: discord.Interaction, attachment: discord.Attachment) -> None:
        # Immediately defer the interaction so Discord never times out during GPU/CPU inference
        await interaction.response.defer(thinking=True)

        if not is_supported_attachment(attachment.filename, attachment.content_type):
            await interaction.followup.send(
                "⚠️ Please upload a supported image (`.png`, `.jpg`, `.webp`) or video clip (`.mp4`, `.mov`, `.webm`)."
            )
            return

        try:
            embed, d_file, hm_path = await bot.analyze_attachment_and_build_reply(attachment)
            try:
                if d_file is not None:
                    await interaction.followup.send(embed=embed, file=d_file)
                else:
                    await interaction.followup.send(embed=embed)
            finally:
                if hm_path and os.path.exists(hm_path):
                    try:
                        os.remove(hm_path)
                    except OSError:
                        pass
        except Exception as exc:
            await interaction.followup.send(f"⚠️ **ZeroFake Inference Error:** `{exc}`")

    return bot


def run_dry_run() -> None:
    """
    Runs a fast (< 5 seconds) CPU dry-run verifying:
      1. Color-coded Discord Embed construction (`0x16a34a` REAL, `0xdc2626` FAKE, `0xd97706` UNKNOWN)
         and `attachment://heatmap.png` image embedding.
      2. End-to-end in-process prediction via `ZeroFakePredictorBridge` (`predict_media_api`) on a
         synthetic video/image media file, confirming all required JSON keys and Grad-CAM heatmap PNG creation.
    """
    from preprocessing.frame_extraction import create_synthetic_video

    start_time = time.time()
    print("[DRY-RUN] Testing discord_bot.py (color-coded embeds, attachment://heatmap.png, and in-process bridge)...")

    # 1. Verify color-coded embeds for REAL, FAKE, and UNKNOWN
    for verdict, expected_color in (
        ("REAL", COLOR_REAL_GREEN),
        ("FAKE", COLOR_FAKE_RED),
        ("UNKNOWN", COLOR_UNKNOWN_AMBER),
    ):
        sample = {
            "verdict": verdict,
            "real_pct": 88.4 if verdict == "REAL" else 11.6,
            "fake_pct": 11.6 if verdict == "REAL" else 88.4,
            "summary": f"Synthetic test summary for {verdict}.",
            "face_status": "Clear face detected (MTCNN conf 99.1%)",
            "heatmap_path": "dummy_heatmap.png",
        }
        emb = build_verdict_embed(sample, filename="sample.png", has_heatmap=True)
        if HAS_DISCORD and discord is not None:
            assert emb.color.value == expected_color, f"Expected {hex(expected_color)} for {verdict}"
            assert emb.image.url == "attachment://heatmap.png", "Embed image URL must be attachment://heatmap.png"
        else:
            assert emb["color"] == expected_color
            assert emb["image_url"] == "attachment://heatmap.png"

    # 2. Verify in-process ZeroFakePredictorBridge on a synthetic face media file
    bridge = ZeroFakePredictorBridge(zerofake_url=None)
    with tempfile.TemporaryDirectory() as tmp_dir:
        media_path = os.path.join(tmp_dir, "test_face.mp4")
        create_synthetic_video(media_path, total_frames=8, height=160, width=160, include_face=True)
        pred_res = bridge.predict(media_path)

        required_keys = {"verdict", "real_pct", "fake_pct", "summary", "face_status", "heatmap_path"}
        assert required_keys.issubset(set(pred_res.keys())), f"Missing keys in prediction result: {pred_res}"
        assert pred_res["verdict"] in ("REAL", "FAKE", "UNKNOWN"), f"Invalid verdict: {pred_res['verdict']}"
        assert pred_res["heatmap_path"] and os.path.exists(pred_res["heatmap_path"]), "Heatmap PNG was not saved!"
        os.remove(pred_res["heatmap_path"])

    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dry-run exceeded 5s limit ({elapsed:.3f}s)"
    print(
        f"[DRY-RUN PASSED] Verified discord_bot.py embeds (0x16a34a / 0xdc2626 / 0xd97706), "
        f"attachment://heatmap.png, and in-process ZeroFake prediction in {elapsed:.3f}s."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the ZeroFake Discord Bot (/check and @ZeroFake mentions).")
    parser.add_argument(
        "--zerofake_url",
        type=str,
        default=None,
        help="Live ZeroFake Gradio URL (overrides ZEROFAKE_URL in .env). Leave unset for local in-process inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/zerofake/best_model.pt",
        help="Local checkpoint path when running in-process without ZEROFAKE_URL.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Discord Bot Token (overrides DISCORD_BOT_TOKEN in .env).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify Discord bot embed formatting and in-process prediction on CPU in under 5 seconds.",
    )
    return parser.parse_args()


def main() -> None:
    if HAS_DOTENV and load_dotenv is not None:
        load_dotenv()

    args = parse_args()
    if args.dry_run:
        run_dry_run()
        return

    token = (args.token or os.environ.get("DISCORD_BOT_TOKEN", "")).strip()
    zerofake_url = (args.zerofake_url or os.environ.get("ZEROFAKE_URL", "")).strip()

    if not token:
        raise RuntimeError(
            "DISCORD_BOT_TOKEN is not set! Create a .env file with DISCORD_BOT_TOKEN=your_token_here "
            "(and optional ZEROFAKE_URL=https://xxxx.gradio.live) or pass --token on the CLI."
        )

    bot = create_discord_bot(
        zerofake_url=zerofake_url if zerofake_url else None,
        checkpoint_path=args.checkpoint,
    )
    print(f"[INFO] Starting ZeroFake Discord Bot ({bot.bridge.mode_description})...")
    bot.run(token)


if __name__ == "__main__":
    main()
