"""分层诊断一个 API 端点到底卡在哪一层。

    python scripts/check_endpoint.py https://modelflare.dev/v1
    python scripts/check_endpoint.py https://api.deepseek.com/v1 --key sk-xxx --model deepseek-chat

依次测 DNS → TCP → TLS 握手 → HTTP 请求 →（可选）真实 chat 调用，
然后给出结论。区分"网络层不通"和"服务端拒绝"很重要：
很多中转网关会**先握手成功、等你的请求发出去再切断**（SSL: UNEXPECTED_EOF），
看起来像 key 错或超时，实际是这个域名在你当前网络下被中断了。
"""
from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

OK = "✓"
NG = "✗"


def step_dns(host: str) -> tuple[bool, str]:
    try:
        ip = socket.gethostbyname(host)
        return True, ip
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def step_tcp(host: str, port: int, timeout: float) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "通"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def step_tls(host: str, port: int, timeout: float) -> tuple[bool, str]:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ss:
                return True, f"{ss.version()}  证书 CN={_cn(ss)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _cn(ss) -> str:
    try:
        cert = ss.getpeercert() or {}
        for rdn in cert.get("subject", ()):
            for k, v in rdn:
                if k == "commonName":
                    return v
    except Exception:  # noqa: BLE001
        pass
    return "?"


def step_http(url: str, timeout: float, headers: dict | None = None
              ) -> tuple[bool, str, int]:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, f"HTTP {r.status}", r.status
    except urllib.error.HTTPError as exc:
        # 401/403/404 都说明网络是通的，只是 key 或路径问题
        return True, f"HTTP {exc.code}（网络通，接口返回错误）", exc.code
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:110]}", 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="例如 https://api.deepseek.com/v1")
    ap.add_argument("--key", default="", help="可选：带上 key 真发一次 chat 请求")
    ap.add_argument("--model", default="", help="可选：要测的模型名")
    ap.add_argument("--timeout", type=float, default=15)
    args = ap.parse_args()

    u = urlparse(args.url)
    host = u.hostname or ""
    port = u.port or (443 if u.scheme == "https" else 80)
    print("=" * 74)
    print(f"端点诊断：{args.url}")
    print("=" * 74)

    results: dict[str, bool] = {}

    ok, info = step_dns(host)
    results["dns"] = ok
    print(f"  1) DNS 解析    {OK if ok else NG}  {info}")
    if not ok:
        print("\n结论：域名解析不了。检查地址是否写错、或本机 DNS 是否正常。")
        return 1

    ok, info = step_tcp(host, port, args.timeout)
    results["tcp"] = ok
    print(f"  2) TCP {port:<5}   {OK if ok else NG}  {info}")
    if not ok:
        print("\n结论：端口连不上。可能是网络/防火墙/代理问题，或该地址不对。")
        return 1

    ok, info = step_tls(host, port, args.timeout) if u.scheme == "https" else (True, "跳过（http）")
    results["tls"] = ok
    print(f"  3) TLS 握手    {OK if ok else NG}  {info}")

    ok, info, code = step_http(args.url.rstrip("/") + "/models", args.timeout)
    results["http"] = ok
    print(f"  4) HTTP 请求   {OK if ok else NG}  {info}")

    print()
    if not results.get("tls"):
        print("结论：TLS 握手就失败了。多半是中间有代理/拦截，或服务端证书有问题。")
        return 1
    if not results.get("http"):
        print("结论：TCP 与 TLS 都正常，但 HTTP 请求发出后连接被切断。")
        print("      这通常意味着**该域名在你当前网络下被中断**（SNI/内容层拦截），")
        print("      或者服务端拒绝非浏览器客户端。不是 key 的问题，也不是超时。")
        print("      → 换一个服务商地址。本机验证过能通的参考端点：")
        print("          https://api.deepseek.com/v1        （国内可直连）")
        print("          https://dashscope.aliyuncs.com/compatible-mode/v1")
        print("      换完再跑一次本脚本确认。")
        return 2
    if code in (401, 403):
        print("结论：网络完全正常，只是这个请求没带（或带错）API Key —— 401/403 属正常现象。")
    elif code == 404:
        print("结论：网络正常，但 /models 路径不存在。有些网关不提供 /models，不影响 chat 调用。")
    else:
        print("结论：网络与接口都正常。")

    if args.key:
        print()
        print("  5) 真实 chat 调用 …")
        body = {"model": args.model or "test",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4}
        req = urllib.request.Request(
            args.url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {args.key}",
                     "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=max(args.timeout, 60)) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            msg = (((data.get("choices") or [{}])[0].get("message") or {})
                   .get("content", ""))
            print(f"       {OK} 调用成功，模型回复：{str(msg)[:40]!r}")
            print("\n结论：这个地址 + key + 模型可以正常出题，放心跑。")
            return 0
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            print(f"       {NG} HTTP {exc.code}: {detail}")
            if exc.code == 401:
                print("\n结论：key 无效或没权限。")
            elif exc.code == 404:
                print(f"\n结论：模型名「{args.model}」在该服务商不存在（换个模型名试试）。")
            else:
                print("\n结论：服务端拒绝了请求，看上面的报错内容。")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"       {NG} {type(exc).__name__}: {exc}")
            print("\n结论：请求发出后被切断/超时。若前面第 4 步是通过的，")
            print("      说明该网关对 chat 长请求会中断——换服务商，或减小请求体。")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
