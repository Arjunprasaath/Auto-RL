"use client";
import dynamic from "next/dynamic";

// Disable SSR — CopilotKit hooks require a browser environment.
const HomePage = dynamic(() => import("@/components/HomePage"), { ssr: false });

export default function Page() {
  return <HomePage />;
}
