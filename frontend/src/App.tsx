import { useState, useCallback, useEffect, type ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import {
  Upload, X, Activity, Newspaper, FileText,
  ShieldAlert, CheckCircle, ChevronDown, ChevronUp, Search, type LucideIcon
} from 'lucide-react'
import './index.css'

type MarketValue = string | number | boolean | null | undefined

interface MarketData {
  [key: string]: MarketValue;
}

interface AnalysisResult {
  companies: string[];
  market_data: Record<string, MarketData>;
  company_scores: Record<string, {
    score?: number;
    confidence?: number;
    stance?: string;
    notes?: string[];
    rank?: number;
    decision_tag?: string;
    breakdown?: {
      growth?: number;
      balance_sheet?: number;
      sentiment?: number;
      data_quality?: number;
    };
  }>;
  news_summary: string;
  document_insights: string;
  risk_analysis: string;
  final_recommendation: string;
  llm_status: string;
  used_fallback: boolean;
}

const getResultNoticeTone = (message: string) => {
  const lowered = message.toLowerCase()
  if (lowered.includes('quota') || lowered.includes('timed out')) return 'warning'
  if (lowered.includes('invalid') || lowered.includes('blocked') || lowered.includes('not available')) return 'danger'
  return 'info'
}

const shouldShowFallbackNotice = (result: AnalysisResult) => {
  if (!result.used_fallback || !result.llm_status) return false
  return !result.news_summary?.trim() || !result.risk_analysis?.trim() || !result.final_recommendation?.trim()
}

// Reusable Collapsible Card Component
interface CollapsibleCardProps {
  title: string;
  icon: LucideIcon;
  children: ReactNode;
  accentColor: string;
}

const CollapsibleCard = ({ title, icon: Icon, children, accentColor }: CollapsibleCardProps) => {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="card"
    >
      <div 
        className="card-header" 
        onClick={() => setIsOpen(!isOpen)}
      >
        <h2 className="card-title" style={{ color: accentColor }}>
          <Icon size={24} /> {title}
        </h2>
        {isOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </div>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="card-content">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const currencySymbols: Record<string, string> = {
  USD: '$',
  EUR: 'EUR ',
  GBP: 'GBP ',
  INR: 'Rs ',
  JPY: 'JPY ',
}

const formatCompactNumber = (value: number) => {
  const absValue = Math.abs(value)
  if (absValue >= 1_000_000_000_000) return `${(value / 1_000_000_000_000).toFixed(2)}T`
  if (absValue >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`
  if (absValue >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (absValue >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  return value.toFixed(2)
}

const formatMetricValue = (key: string, value: MarketValue, currencyCode?: string) => {
  if (value === null || value === undefined || value === '' || value === 'Data Not Available') {
    return 'Data Not Available'
  }

  if (typeof value !== 'number') {
    return String(value)
  }

  const symbol = currencySymbols[currencyCode ?? ''] ?? `${currencyCode ?? ''} `

  if (key === 'currentPrice') {
    return `${symbol}${value.toFixed(2)}`
  }

  if (key === 'marketCap' || key === 'revenue' || key === 'ebitda') {
    return `${symbol}${formatCompactNumber(value)}`
  }

  if (key === 'revenueGrowth') {
    return `${(value * 100).toFixed(1)}%`
  }

  if (key === 'debtToEquity') {
    return `${value.toFixed(2)}`
  }

  return String(value)
}

function App() {
  const [query, setQuery] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState('');

  // Step-based loading states simulation
  useEffect(() => {
    if (!loading) return;
    const steps = [
      "Extracting companies...",
      "Resolving financial tickers...",
      "Fetching real-time market data...",
      "Analyzing latest news & sentiment...",
      "Generating final insights..."
    ];
    let i = 0;
    setLoadingStep(steps[i]);
    const interval = setInterval(() => {
      i = Math.min(i + 1, steps.length - 1);
      setLoadingStep(steps[i]);
    }, 2500);
    return () => clearInterval(interval);
  }, [loading]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles]);
  }, []);

  const removeFile = (name: string) => {
    setFiles(files.filter(f => f.name !== name));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: { 'application/pdf': ['.pdf'], 'text/plain': ['.txt'] }
  });

  const handleAnalyze = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const formData = new FormData();
      formData.append('query', query);
      
      if (files.length > 0) {
        files.forEach(file => formData.append('documents', file));
      } else {
        formData.append('documents', new Blob([''], { type: 'application/octet-stream' }), '');
      }

      const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || ('Analysis request failed: ' + response.statusText));
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'An error occurred during analysis.'
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const renderMarketData = (marketData: Record<string, MarketData>) => {
    if (!marketData || Object.keys(marketData).length === 0) return <p>No market data available.</p>;
    
    return Object.entries(marketData).map(([company, data]) => {
      const metrics = Object.entries(data)
        .filter(([k]) => k !== 'symbol' && k !== 'companyName' && k !== 'currency' && !k.startsWith('_'))
        .map(([key, val]) => ({
          key,
          formattedValue: formatMetricValue(key, val, typeof data.currency === 'string' ? data.currency : undefined),
        }))

      const availableMetrics = metrics.filter(({ formattedValue }) => formattedValue !== 'Data Not Available')
      const unavailableCount = metrics.length - availableMetrics.length
      const hasOnlyPrice = availableMetrics.length === 1 && availableMetrics[0].key === 'currentPrice'

      return (
        <div key={company} className="market-data-company">
          <h3 style={{ color: 'var(--text-main)', marginTop: 0, marginBottom: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
            {company} {data.symbol ? `(${data.symbol})` : ''}
          </h3>

          {availableMetrics.length > 0 ? (
            <div className="market-data-grid">
              {availableMetrics.map(({ key, formattedValue }) => (
                <div className="market-metric" key={key}>
                  <span className="metric-label">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                  <span className="metric-value">{formattedValue}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="market-data-note">
              Detailed market fundamentals are not available right now for {company}.
            </div>
          )}

          {(hasOnlyPrice || unavailableCount > 0) && (
            <div className="market-data-note">
              {hasOnlyPrice
                ? `The live market feed returned only quote data for ${company} right now, so fundamentals like revenue, EBITDA, and sector are missing.`
                : `Some live market fields are temporarily unavailable for ${company}.`}
            </div>
          )}
        </div>
      )
    });
  }

  const renderCompanyScores = (companyScores: AnalysisResult['company_scores']) => {
    if (!companyScores || Object.keys(companyScores).length === 0) return <p>No scorecard available.</p>;

    return Object.entries(companyScores).map(([company, scorecard]) => (
      <div key={company} className="market-data-company">
        <h3 style={{ color: 'var(--text-main)', marginTop: 0, marginBottom: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
          {company}
        </h3>
        <div className="market-data-grid">
          <div className="market-metric">
            <span className="metric-label">Rank</span>
            <span className="metric-value">{typeof scorecard.rank === 'number' ? `#${scorecard.rank}` : 'N/A'}</span>
          </div>
          <div className="market-metric">
            <span className="metric-label">Score</span>
            <span className="metric-value">{typeof scorecard.score === 'number' ? `${scorecard.score}/100` : 'N/A'}</span>
          </div>
          <div className="market-metric">
            <span className="metric-label">Confidence</span>
            <span className="metric-value">{typeof scorecard.confidence === 'number' ? `${scorecard.confidence}%` : 'N/A'}</span>
          </div>
          <div className="market-metric">
            <span className="metric-label">Stance</span>
            <span className="metric-value">{scorecard.stance ?? 'N/A'}</span>
          </div>
          <div className="market-metric">
            <span className="metric-label">Decision Tag</span>
            <span className="metric-value">{scorecard.decision_tag ?? 'N/A'}</span>
          </div>
        </div>
        {scorecard.breakdown && (
          <div className="market-data-grid" style={{ marginTop: '1rem' }}>
            <div className="market-metric">
              <span className="metric-label">Growth</span>
              <span className="metric-value">{typeof scorecard.breakdown.growth === 'number' ? `${scorecard.breakdown.growth}/40` : 'N/A'}</span>
            </div>
            <div className="market-metric">
              <span className="metric-label">Balance Sheet</span>
              <span className="metric-value">{typeof scorecard.breakdown.balance_sheet === 'number' ? `${scorecard.breakdown.balance_sheet}/20` : 'N/A'}</span>
            </div>
            <div className="market-metric">
              <span className="metric-label">Sentiment</span>
              <span className="metric-value">{typeof scorecard.breakdown.sentiment === 'number' ? `${scorecard.breakdown.sentiment}/20` : 'N/A'}</span>
            </div>
            <div className="market-metric">
              <span className="metric-label">Data Quality</span>
              <span className="metric-value">{typeof scorecard.breakdown.data_quality === 'number' ? `${scorecard.breakdown.data_quality}/20` : 'N/A'}</span>
            </div>
          </div>
        )}
        {Array.isArray(scorecard.notes) && scorecard.notes.length > 0 && (
          <div className="market-data-note">
            {scorecard.notes.join('; ')}.
          </div>
        )}
      </div>
    ))
  }

  return (
    <div className="app-container">
      <header>
        <motion.h1 
          initial={{ opacity: 0, y: -20 }} 
          animate={{ opacity: 1, y: 0 }}
        >
          InvestIQ AI
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          AI-Powered Investment Intelligence
        </motion.p>
      </header>

      <motion.div 
        className="search-section"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
      >
        <textarea
          className="search-input"
          placeholder="e.g. Analyze Nvidia vs AMD hardware revenue growth over the next year focusing on AI market risks."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          <Upload size={32} color={isDragActive ? "var(--accent)" : "var(--text-muted)"} />
          <p>{isDragActive ? "Drop files here" : "Drag & drop financial PDFs or click to upload context"}</p>
        </div>

        {files.length > 0 && (
          <div className="dropdown-files">
            {files.map(f => (
              <div key={f.name} className="file-item">
                <span>📄 {f.name}</span>
                <X size={16} style={{ cursor: 'pointer', color: 'var(--danger)' }} onClick={() => removeFile(f.name)} />
              </div>
            ))}
          </div>
        )}
        
        <button 
          className="analyze-button" 
          onClick={handleAnalyze}
          disabled={loading || !query.trim()}
        >
          <Search size={20} />
          {loading ? 'Processing...' : 'Run Analysis'}
        </button>
      </motion.div>

      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ color: 'var(--danger)', marginBottom: '2rem', textAlign: 'center', fontWeight: 'bold' }}>
          {error}
        </motion.div>
      )}

      {loading && (
        <motion.div 
          className="loading-container"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="spinner"></div>
          <motion.div 
            key={loadingStep}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="loading-text"
          >
            {loadingStep}
          </motion.div>
        </motion.div>
      )}

      {result && !loading && (
        <div className="results-container">
          {shouldShowFallbackNotice(result) && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className={`result-notice result-notice-${getResultNoticeTone(result.llm_status)}`}
            >
              {result.llm_status} The analysis below uses grounded fallback logic from market data, news, and documents.
            </motion.div>
          )}

          {result.companies && result.companies.length > 0 && (
            <motion.div 
              className="entities-container"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <strong style={{ alignSelf: 'center', color: 'var(--text-muted)' }}>Analyzed Entities:</strong>
              {result.companies.map((c, i) => (
                <span key={i} className="tag">{c}</span>
              ))}
            </motion.div>
          )}

          {result.market_data && Object.keys(result.market_data).length > 0 && (
            <CollapsibleCard title="Market Data" icon={Activity} accentColor="#60a5fa">
              {renderMarketData(result.market_data)}
            </CollapsibleCard>
          )}

          {result.company_scores && Object.keys(result.company_scores).length > 0 && (
            <CollapsibleCard title="Scorecard" icon={Activity} accentColor="#34d399">
              {renderCompanyScores(result.company_scores)}
            </CollapsibleCard>
          )}

          {result.news_summary && (
            <CollapsibleCard title="News Summary" icon={Newspaper} accentColor="#a78bfa">
              <div className="prose">{result.news_summary}</div>
            </CollapsibleCard>
          )}

          {result.document_insights && !result.document_insights.includes("Parsing or retrieval failure") && !result.document_insights.includes("No documents") && (
            <CollapsibleCard title="Document Insights" icon={FileText} accentColor="#f472b6">
              <div className="prose">{result.document_insights}</div>
            </CollapsibleCard>
          )}

          {result.risk_analysis && (
            <CollapsibleCard title="Risk Analysis" icon={ShieldAlert} accentColor="var(--warning)">
              <div className="prose">{result.risk_analysis}</div>
            </CollapsibleCard>
          )}

          {result.final_recommendation && (
            <CollapsibleCard title="Final Recommendation" icon={CheckCircle} accentColor="var(--success)">
              <div className="prose" style={{ color: 'var(--text-main)', fontSize: '1.05rem' }}>
                {result.final_recommendation}
              </div>
            </CollapsibleCard>
          )}
        </div>
      )}
    </div>
  )
}

export default App
