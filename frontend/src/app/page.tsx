'use client';

import { useEffect, useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Chart = dynamic(() => import('react-apexcharts'), { ssr: false });

type OptionsResp = {
  makes: string[];
  models: string[];      // global model list (fallback)
  bodies: string[];
  year_min: number;
  year_max: number;
};

type SummaryResp = {
  rows: number;
  median_price: number;
  median_mileage: number;
  unique_makes: number;
  unique_models: number;
};

type ChartsResp = {
  price_hist: { bins: number[]; counts: number[] };
  price_by_year: { year: number[]; price: number[] };
  top_models: { model: string[]; price: number[] };
  make_share: { make: string[]; count: number[] };
  model_share: { model: string[]; count: number[] };
};

// ---------- Config ----------
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

// Neon green on black theme
const ACCENT = '#7CFC00';
const ACCENT_SOFT = '#b7ff5c';
const CARD_BG = 'rgba(255,255,255,0.04)';

function num(n?: number | null) {
  if (n == null) return '—';
  return n.toLocaleString();
}
function usd(n?: number | null) {
  if (n == null) return '—';
  return `$${Math.round(n).toLocaleString()}`;
}

// ---------- UI ----------
export default function Page() {
  const [options, setOptions] = useState<OptionsResp | null>(null);
  const [make, setMake] = useState<string>('(all)');
  const [modelsForMake, setModelsForMake] = useState<string[]>([]);
  const [model, setModel] = useState<string>('(all)');
  const [y0, setY0] = useState<number | null>(null);
  const [y1, setY1] = useState<number | null>(null);

  const [summary, setSummary] = useState<SummaryResp | null>(null);
  const [charts, setCharts] = useState<ChartsResp | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [err, setErr] = useState<string | null>(null);

  // Load options on mount
  useEffect(() => {
    (async () => {
      try {
        setErr(null);
        const r = await fetch(`${API_BASE}/options`, { cache: 'no-store' });
        const js = (await r.json()) as OptionsResp;
        setOptions(js);
        setY0(js.year_min);
        setY1(js.year_max);
      } catch (e: any) {
        setErr(`Options error: ${e?.message || e}`);
      }
    })();
  }, []);

  // When Make changes, refresh the Model list using /charts model_share (so it’s truly per-make)
  useEffect(() => {
    (async () => {
      if (!options) return;
      if (make === '(all)') {
        setModelsForMake([]);
        setModel('(all)');
        return;
      }
      try {
        const url = new URL(`${API_BASE}/charts`);
        url.searchParams.set('make', make);
        if (y0 != null) url.searchParams.set('y0', String(y0));
        if (y1 != null) url.searchParams.set('y1', String(y1));
        const r = await fetch(url.toString(), { cache: 'no-store' });
        const js = (await r.json()) as ChartsResp;
        const list = js?.model_share?.model || [];
        setModelsForMake(list);
        setModel('(all)'); // reset selection when make changes
      } catch {
        // fallback to global models if charts call fails
        setModelsForMake(options.models || []);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [make, y0, y1, options]);

  // Load summary + charts whenever filters change
  useEffect(() => {
    if (!options || y0 == null || y1 == null) return;
    (async () => {
      try {
        setLoading(true);
        setErr(null);

        // summary
        const sUrl = new URL(`${API_BASE}/summary`);
        if (make !== '(all)') sUrl.searchParams.set('make', make);
        if (model !== '(all)') sUrl.searchParams.set('model', model);
        sUrl.searchParams.set('y0', String(y0));
        sUrl.searchParams.set('y1', String(y1));
        const sr = await fetch(sUrl.toString(), { cache: 'no-store' });
        const sJson = (await sr.json()) as SummaryResp;
        setSummary(sJson);

        // charts
        const cUrl = new URL(`${API_BASE}/charts`);
        if (make !== '(all)') cUrl.searchParams.set('make', make);
        if (model !== '(all)') cUrl.searchParams.set('model', model);
        cUrl.searchParams.set('y0', String(y0));
        cUrl.searchParams.set('y1', String(y1));
        const cr = await fetch(cUrl.toString(), { cache: 'no-store' });
        const cJson = (await cr.json()) as ChartsResp;
        setCharts(cJson);
      } catch (e: any) {
        setErr(`Data error: ${e?.message || e}`);
      } finally {
        setLoading(false);
      }
    })();
  }, [options, make, model, y0, y1]);

  const modelOptions = useMemo<string[]>(
    () => (make === '(all)' ? [] : modelsForMake),
    [make, modelsForMake]
  );

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="mx-auto max-w-7xl px-5 py-6">
        {/* Header */}
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight">
              Used Car Price — <span className="text-[var(--accent,#7CFC00)]">Dashboard</span>
            </h1>
            <p className="text-gray-400 mt-1">
              Interactive analytics powered by FastAPI + ApexCharts.
            </p>
          </div>
          <Badge />
        </div>

        {/* Filters */}
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-4">
          {/* Make */}
          <div className="card">
            <label className="label">Make</label>
            <select
              value={make}
              onChange={(e) => setMake(e.target.value)}
              className="select"
            >
              <option>(all)</option>
              {options?.makes?.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* Model (cascades) */}
          <div className="card">
            <label className="label">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="select"
              disabled={make === '(all)' || modelOptions.length === 0}
            >
              <option>(all)</option>
              {modelOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            {make !== '(all)' && (
              <p className="text-xs text-gray-400 mt-2">
                {modelOptions.length.toLocaleString()} models for {make}
              </p>
            )}
          </div>

          {/* Year range */}
          <div className="card">
            <label className="label">Year (from)</label>
            <input
              type="number"
              className="input"
              value={y0 ?? ''}
              min={options?.year_min}
              max={y1 ?? options?.year_max}
              onChange={(e) => setY0(Number(e.target.value))}
            />
          </div>
          <div className="card">
            <label className="label">Year (to)</label>
            <input
              type="number"
              className="input"
              value={y1 ?? ''}
              min={y0 ?? options?.year_min}
              max={options?.year_max}
              onChange={(e) => setY1(Number(e.target.value))}
            />
          </div>
        </div>

        {/* KPIs */}
        <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <KPI title="Listings" value={num(summary?.rows)} />
          <KPI title="Median Price" value={usd(summary?.median_price)} />
          <KPI title="Median Mileage" value={`${num(summary?.median_mileage)} mi`} />
          <KPI
            title="Distinct makes / models"
            value={`${num(summary?.unique_makes)} / ${num(summary?.unique_models)}`}
          />
        </div>

        {/* Errors */}
        {err && (
          <div className="mt-4 rounded-lg border border-red-500 bg-red-500/10 px-4 py-3 text-sm">
            {err}
          </div>
        )}

        {/* Charts */}
        <div className="mt-6 grid grid-cols-1 gap-5 lg:grid-cols-2">
          {/* Price distribution */}
          <div className="card">
            <div className="section-title mb-3">Price distribution</div>
            <Chart
              type="bar"
              height={320}
              series={[
                {
                  name: 'Listings',
                  data: charts?.price_hist?.counts || [],
                },
              ]}
              options={{
                chart: { background: 'transparent', toolbar: { show: false } },
                theme: { mode: 'dark' },
                plotOptions: { bar: { columnWidth: '75%', borderRadius: 3 } },
                dataLabels: { enabled: false },
                xaxis: {
                  categories: charts?.price_hist?.bins?.slice(1) || [],
                  labels: { rotate: 0, formatter: (v: any) => `$${Number(v).toLocaleString()}` },
                },
                yaxis: { labels: { formatter: (v: any) => Number(v).toLocaleString() } },
                colors: [ACCENT],
                grid: { borderColor: 'rgba(255,255,255,0.1)' },
              }}
            />
          </div>

          {/* Median price by year */}
          <div className="card">
            <div className="section-title mb-3">Median price by year</div>
            <Chart
              type="area"
              height={320}
              series={[
                {
                  name: 'Median price',
                  data:
                    charts?.price_by_year?.year?.map((y, i) => ({
                      x: y,
                      y: charts?.price_by_year?.price?.[i],
                    })) || [],
                },
              ]}
              options={{
                chart: { background: 'transparent', toolbar: { show: false } },
                theme: { mode: 'dark' },
                dataLabels: { enabled: false },
                stroke: { width: 2, curve: 'smooth' },
                xaxis: { labels: { formatter: (v: any) => `${v}` } },
                yaxis: { labels: { formatter: (v: any) => `$${Number(v).toLocaleString()}` } },
                colors: [ACCENT],
                fill: { type: 'gradient', gradient: { shadeIntensity: 0.8, opacityFrom: 0.25, opacityTo: 0.05 } },
                grid: { borderColor: 'rgba(255,255,255,0.1)' },
              }}
            />
          </div>

          {/* Top models (median price) */}
          <div className="card">
            <div className="section-title mb-3">Top 10 models (median price)</div>
            <Chart
              type="bar"
              height={360}
              series={[
                {
                  name: 'Median price',
                  data: (charts?.top_models?.model || []).map((label, i) => ({
                    x: label,
                    y: charts?.top_models?.price?.[i],
                  })),
                },
              ]}
              options={{
                chart: { background: 'transparent', toolbar: { show: false } },
                theme: { mode: 'dark' },
                plotOptions: { bar: { horizontal: true, barHeight: '70%', borderRadius: 3 } },
                dataLabels: { enabled: false },
                xaxis: { labels: { formatter: (v: any) => `$${Number(v).toLocaleString()}` } },
                colors: [ACCENT],
                grid: { borderColor: 'rgba(255,255,255,0.1)' },
              }}
            />
          </div>

          {/* Make share (donut) */}
          <div className="card">
            <div className="section-title mb-3">Make share</div>
            <Chart
              type="donut"
              height={320}
              series={charts?.make_share?.count || []}
              options={{
                chart: { background: 'transparent' },
                labels: charts?.make_share?.make || [],
                legend: { position: 'bottom' },
                theme: { mode: 'dark' },
                dataLabels: { enabled: false },
                stroke: { width: 0 },
                plotOptions: { pie: { donut: { size: '65%' } } },
                colors: [ACCENT, ACCENT_SOFT, '#dbffb6', '#eaffd6', '#a7ff83', '#6cff00'],
              }}
            />
          </div>

          {/* Model share (donut) */}
          <div className="card">
            <div className="section-title mb-3">Model share</div>
            <Chart
              type="donut"
              height={320}
              series={charts?.model_share?.count || []}
              options={{
                chart: { background: 'transparent' },
                labels: charts?.model_share?.model || [],
                legend: { position: 'bottom' },
                theme: { mode: 'dark' },
                dataLabels: { enabled: false },
                stroke: { width: 0 },
                plotOptions: { pie: { donut: { size: '65%' } } },
                colors: [ACCENT, ACCENT_SOFT, '#dbffb6', '#eaffd6', '#a7ff83', '#6cff00'],
              }}
            />
          </div>
        </div>

        {/* Loading overlay */}
        {loading && (
          <div className="fixed inset-x-0 bottom-6 flex justify-center">
            <div className="rounded-full bg-white/5 px-4 py-2 text-sm backdrop-blur">
              Loading…
            </div>
          </div>
        )}
      </div>

      {/* Local styles */}
      <style jsx global>{`
        :root { --accent: ${ACCENT}; }
        .card {
          background: ${CARD_BG};
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 16px;
          padding: 16px;
        }
        .label {
          display: block;
          font-size: 0.82rem;
          color: #9ca3af;
          margin-bottom: 6px;
        }
        .select, .input {
          width: 100%;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 10px;
          padding: 10px 12px;
          outline: none;
        }
        .select:focus, .input:focus {
          border-color: ${ACCENT};
          box-shadow: 0 0 0 3px ${ACCENT}22;
        }
        .section-title {
          font-weight: 800;
          letter-spacing: .2px;
        }
      `}</style>
    </div>
  );
}

// ---------- Small components ----------
function KPI({ title, value }: { title: string; value: string }) {
  return (
    <div className="card">
      <div className="text-gray-400 text-xs font-semibold">{title}</div>
      <div className="mt-1 text-2xl font-extrabold tracking-tight">{value}</div>
    </div>
  );
}

function Badge() {
  return (
    <div className="inline-flex select-none items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-gray-300">
      API: <span className="font-semibold text-[var(--accent,#7CFC00)]">FastAPI</span>
      <span className="mx-1">•</span>
      UI: <span className="font-semibold">Next + ApexCharts</span>
    </div>
  );
}