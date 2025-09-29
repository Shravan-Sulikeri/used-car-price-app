import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Used Car Price — Dashboard",
  description: "Interactive analytics and price prediction",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-black text-zinc-100">{children}</body>
    </html>
  );
}