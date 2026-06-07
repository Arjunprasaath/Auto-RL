import type { NextConfig } from "next";
import fs from "fs";
import path from "path";

// Load the shared autorl/.env (one level above ui/) so this app needs no
// separate .env.local file — all keys live in one place.
const sharedEnv = path.resolve(__dirname, "../.env");
if (fs.existsSync(sharedEnv)) {
  for (const line of fs.readFileSync(sharedEnv, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    const val = trimmed.slice(eq + 1).trim();
    // Never overwrite vars already set in the shell environment
    if (key && !(key in process.env)) process.env[key] = val;
  }
}

const nextConfig: NextConfig = {};

export default nextConfig;
