"use client";

import { useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Options = {
  makes: string[];
  models: string[];
  models_by_make: Record<string, string[]>;
  bodies: string[];
  year_min: number | null;
  year_max: number | null;
};

export default function Home() {
  const [opts, setOpts] = useState<Options | null>(null);
  const [make, setMake] = useState<string>("");
  const [model, setModel] = useState<string>("");
  const [body, setBody] = useState<string>("");
  const [year, setYear] = useState<number>(2020);
  const [mileage, setMileage] = useState<number>(45000);
  const [price, setPrice] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/options`).then(r => r.json()).then(setOpts).catch(e => setErr(String(e)));
  }, []);

  // dependent models by make
  const modelOptions = useMemo(() => {
    if (!opts) return [];
    if (make && opts.models_by_make?.[make]) return opts.models_by_make[make];
    return opts.models || [];
  }, [opts, make]);

  // ensure model stays valid when make changes
  useEffect(() => {
    if (model && !modelOptions.includes(model)) setModel("");
  }, [modelOptions, model]);

  async function predict() {
    setLoading(true); setErr(null); setPrice(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST", headers: {"content-type":"application/json"},
        body: JSON.stringify({
          year, mileage,
          make: make || "",
          model: model || "",
          body: body || null
        })
      });
      const j = await res.json();
      if (!j.ok) throw new Error(j.error || "Prediction failed");
      setPrice(j.price_usd);
    } catch (e:any) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-black text-white">
      <div className="max-w-5xl mx-auto p-6">
        <h1 className="text-3xl font-extrabold mb-2">Used Car Price — Predictor</h1>
        <p className="text-neutral-400 mb-8">Backed by FastAPI + GBM model.</p>

        {!opts ? (
          <div className="text-neutral-400">Loading options…</div>
        ) : (
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block mb-1 text-sm text-neutral-400">Make</label>
                <select className="w-full bg-neutral-900 border border-neutral-800 rounded px-3 py-2"
                        value={make} onChange={e=>setMake(e.target.value)}>
                  <option value="">(choose)</option>
                  {opts.makes.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              <div>
                <label className="block mb-1 text-sm text-neutral-400">Model</label>
                <select className="w-full bg-neutral-900 border border-neutral-800 rounded px-3 py-2"
                        value={model} onChange={e=>setModel(e.target.value)}>
                  <option value="">(choose)</option>
                  {modelOptions.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block mb-1 text-sm text-neutral-400">Year</label>
                  <input type="number" className="w-full bg-neutral-900 border border-neutral-800 rounded px-3 py-2"
                         value={year} onChange={e=>setYear(Number(e.target.value))}/>
                </div>
                <div>
                  <label className="block mb-1 text-sm text-neutral-400">Mileage</label>
                  <input type="number" className="w-full bg-neutral-900 border border-neutral-800 rounded px-3 py-2"
                         value={mileage} onChange={e=>setMileage(Number(e.target.value))}/>
                </div>
              </div>

              <div>
                <label className="block mb-1 text-sm text-neutral-400">Body (optional)</label>
                <select className="w-full bg-neutral-900 border border-neutral-800 rounded px-3 py-2"
                        value={body} onChange={e=>setBody(e.target.value)}>
                  <option value="">(none)</option>
                  {opts.bodies.map(b => <option key={b} value={b}>{b}</option>)}
                </select>
              </div>

              <button onClick={predict}
                      className="mt-2 inline-flex items-center justify-center px-4 py-2 rounded bg-lime-400 text-black font-semibold hover:bg-lime-300 disabled:opacity-60"
                      disabled={loading}>
                {loading ? "Predicting…" : "Predict"}
              </button>

              {err && <div className="text-red-400 text-sm">{err}</div>}
            </div>

            <div className="flex items-center justify-center">
              <div className="w-full rounded-2xl border border-neutral-800 p-8 bg-neutral-950 text-center">
                <div className="text-neutral-400 mb-2">Estimated Price</div>
                <div className="text-5xl font-extrabold tracking-tight">
                  {price === null ? "—" : `$${Math.round(price).toLocaleString()}`}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}