#!/usr/bin/env bash
# Exercises the npm launcher the way a broken npx cache leaves it: the main
# package installed with no platform package. The launcher must fetch the
# platform binary from the registry, verify it, and run it; and it must refuse a
# tarball that fails the integrity check.
#   scripts/test-launcher.sh [version]   (default: latest published)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
version="${1:-$(npm view mentedb-mcp version)}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
pkg="$work/app/node_modules/mentedb-mcp"
mkdir -p "$pkg" "$work/home"
cp -r "$here/npm/mentedb-mcp/bin" "$pkg/"
node -e '
const fs = require("fs");
const [src, dst, v] = process.argv.slice(1);
const d = JSON.parse(fs.readFileSync(src, "utf8"));
d.version = v;
for (const k of Object.keys(d.optionalDependencies)) d.optionalDependencies[k] = v;
fs.writeFileSync(dst, JSON.stringify(d));
' "$here/npm/mentedb-mcp/package.json" "$pkg/package.json" "$version"

echo "launcher self heal against mentedb-mcp $version"
HOME="$work/home" USERPROFILE="$work/home" node "$pkg/bin/mentedb-mcp" --help | grep -q "MCP server for MenteDB"
echo "ok: fetched, verified and ran the platform binary"
HOME="$work/home" USERPROFILE="$work/home" node "$pkg/bin/mentedb-mcp" --help >/dev/null
echo "ok: reused the cached binary"

rm -rf "$work/home/.mentedb"
if LAUNCHER="$pkg/bin/mentedb-mcp" HOME="$work/home" USERPROFILE="$work/home" node -e '
const real = global.fetch;
global.fetch = async (u) => {
  const r = await real(u);
  if (!String(u).endsWith(".tgz")) return r;
  const b = Buffer.from(await r.arrayBuffer());
  b[100] ^= 1;
  return new Response(b);
};
process.argv = ["node", "launcher", "--help"];
require(process.env.LAUNCHER);
' 2>/dev/null; then
  echo "FAIL: a tampered tarball was accepted"; exit 1
fi
echo "ok: tampered tarball refused"
