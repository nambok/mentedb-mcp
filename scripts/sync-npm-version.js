#!/usr/bin/env node

// Reads the version from Cargo.toml and updates all npm/*/package.json files.
const fs = require("fs");
const path = require("path");

const cargoToml = fs.readFileSync(
  path.join(__dirname, "..", "Cargo.toml"),
  "utf8"
);
const match = cargoToml.match(/^version\s*=\s*"([^"]+)"/m);
if (!match) {
  console.error("Could not find version in Cargo.toml");
  process.exit(1);
}
const version = match[1];

const npmDir = path.join(__dirname, "..", "npm");
const packages = fs.readdirSync(npmDir);

for (const pkg of packages) {
  const pkgJsonPath = path.join(npmDir, pkg, "package.json");
  if (!fs.existsSync(pkgJsonPath)) continue;

  const pkgJson = JSON.parse(fs.readFileSync(pkgJsonPath, "utf8"));
  pkgJson.version = version;

  if (pkgJson.optionalDependencies) {
    for (const dep of Object.keys(pkgJson.optionalDependencies)) {
      pkgJson.optionalDependencies[dep] = version;
    }
  }

  fs.writeFileSync(pkgJsonPath, JSON.stringify(pkgJson, null, 2) + "\n");
  console.log(`Updated ${pkg} to ${version}`);
}
