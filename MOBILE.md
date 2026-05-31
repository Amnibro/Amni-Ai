# Adam on your phone 📱

Adam itself runs on your PC (the GF(17) bake needs a GPU), so your phone is a **thin client**: it talks to the PC-hosted server. The web UI is already responsive and **installable as an app** (PWA), so this is mostly two things — a **secure tunnel** and an **access token**.

> ⚠️ Adam can run shell commands, read/write files, and operate the PC. **Never expose it to the internet without the token gate.** The steps below turn that gate on automatically.

## Quick start (Cloudflare tunnel + token)

From the repo, one command:

```bash
python scripts/tunnel.py --serve            # Windows
python3 scripts/tunnel.py --serve           # Mac/Linux
```

It will:
1. create a persistent access token at `~/.amni-ai/mobile_token.txt`,
2. launch Adam **with that token** (so every API call requires it),
3. start `cloudflared` and print a public `https://….trycloudflare.com` URL,
4. show a **one-tap link** `https://…/?token=…` (and a QR if you `pip install qrcode`).

On the phone: **open that link once** → the token is saved on-device and stripped from the URL → tap your browser's **"Add to Home Screen"** → Adam now opens fullscreen, like a native app.

Already running Adam yourself? Drop `--serve` and the script prints the exact `AMNI_AUTH_TOKEN_FILE=… serve …` command to start it with the matching token.

`cloudflared` missing? `winget install Cloudflare.cloudflared` (Windows) · `brew install cloudflared` (Mac) · https://pkg.cloudflare.com (Linux). Rotate the token anytime with `--new-token`.

## How the gate works
- Set `AMNI_AUTH_TOKEN` (or `AMNI_AUTH_TOKEN_FILE`) on the server → the gate turns **on**. Unset → **off** (fine for localhost).
- The app **shell** (`/`, `/unified`, `/jarvis`, `/healthz`, manifest, service worker, icons) loads without the token; **every API** (`/chat`, skills, shell, file, PC ops, `/update`) requires it.
- The phone sends the token as an `X-Amni-Token` header on every request (auto-attached by the UI). It accepts the token via `?token=`, the `X-Amni-Token` header, a `Bearer` token, or an `amni_token` cookie.

## Alternatives
- **Tailscale (most private):** put phone + PC on a Tailscale mesh, then open `http://<pc-tailscale-ip>:7700` — no public exposure at all. You can still set a token for defense-in-depth.
- **Home Wi-Fi only:** `http://<pc-lan-ip>:7700` — zero setup, same network only.
- **Native Android app:** a thin WebView wrapper around the tunnel URL + FCM push is on the roadmap (same toolchain as Amni-Haven / Amni-Learn).

## Requirements
- The PC must be **on and running Adam** for the phone to work (it's a remote client, not on-device inference).
- The PWA caches only the static shell (icons/css/js) for speed — never your conversations or API responses.
