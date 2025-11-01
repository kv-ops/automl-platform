import { useState } from 'react';

export default function AxellePreLaunch() {
  const [email, setEmail] = useState('');
  const [linkedin, setLinkedin] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Submitting:', { email, linkedin, to: 'kv@axelle.ai' });
    setSubmitted(true);
    setTimeout(() => {
      setEmail('');
      setLinkedin('');
      setSubmitted(false);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans scroll-smooth">
      {/* Top Navigation Bar */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-sm border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <img src="/axelle-logo.png" alt="Axelle AI" className="w-8 h-8" />
            <span className="text-xl font-bold tracking-tight">AXELLE AI</span>
          </div>
          <div className="hidden md:flex space-x-8 text-sm">
            <a href="#" className="hover:text-gray-300 transition">Home</a>
            <a href="#features" className="hover:text-gray-300 transition">Features</a>
            <a href="#" className="hover:text-gray-300 transition">Integrations</a>
          </div>
        </div>
      </nav>

      {/* Hero Section with HUD-style overlay */}
      <section id="hero" className="relative min-h-screen flex items-center justify-center pt-20">
        {/* Background Grid */}
        <div className="absolute inset-0 opacity-20">
          <div className="h-full w-full" style={{
            backgroundImage: 'linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px)',
            backgroundSize: '50px 50px'
          }}></div>
        </div>

        {/* Corner Indicators - HUD Style */}
        <div className="absolute top-24 left-6 w-16 h-16 border-l-2 border-t-2 border-gray-700"></div>
        <div className="absolute top-24 right-6 w-16 h-16 border-r-2 border-t-2 border-gray-700"></div>
        <div className="absolute bottom-6 left-6 w-16 h-16 border-l-2 border-b-2 border-gray-700"></div>
        <div className="absolute bottom-6 right-6 w-16 h-16 border-r-2 border-b-2 border-gray-700"></div>

        {/* Main Content */}
        <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
          <h1 className="text-6xl md:text-7xl font-extrabold mb-6 leading-tight tracking-tight">
            AI-Powered AutoML
            <br />
            <span className="text-gray-400">for Data Science</span>
          </h1>
          <p className="text-xl text-gray-400 mb-12 max-w-3xl mx-auto font-normal">
            We democratize data science by turning your data into value with AI&nbsp;agents — accessible, no-code and ready-to-use.
          </p>

          {/* Waitlist Form */}
          <form onSubmit={handleSubmit} className="max-w-2xl mx-auto mb-8">
            <div className="flex flex-col md:flex-row gap-4 mb-6">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
                required
                className="flex-1 px-6 py-4 bg-gray-900/50 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-white transition"
              />
              <input
                type="url"
                value={linkedin}
                onChange={(e) => setLinkedin(e.target.value)}
                placeholder="LinkedIn profile (optional)"
                className="flex-1 px-6 py-4 bg-gray-900/50 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-white transition"
              />
            </div>
            <button
              type="submit"
              disabled={submitted}
              className="px-8 py-3 bg-white text-black font-medium hover:bg-gray-200 transition disabled:opacity-50"
            >
              {submitted ? 'Welcome aboard! 🚀' : 'Join Waitlist'}
            </button>
          </form>

          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
            <span><span className="text-white font-medium">75</span> professionals already joined</span>
          </div>
        </div>
      </section>

      {/* Features Section with Quadrant Layout */}
      <section id="features" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold mb-16 tracking-tight">Developed in France, Axelle AI embodies<br />the triple role of an&nbsp;all-in-one virtual data expert.</h2>

          <div className="grid md:grid-cols-3 gap-1 bg-gray-800">
            {/* Consultant */}
            <div className="bg-black p-12 border border-gray-800 hover:border-gray-600 transition group">
              <div className="mb-6">
                <svg className="w-12 h-12 text-gray-400 group-hover:text-white transition" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 group-hover:text-gray-300 transition">Consultant</h3>
              <p className="text-sm text-gray-400 mb-4 font-normal">
                Identifies your business KPIs and structures your dataset with AI agents.
              </p>
            </div>

            {/* Scientist */}
            <div className="bg-black p-12 border border-gray-800 hover:border-gray-600 transition group">
              <div className="mb-6">
                <svg className="w-12 h-12 text-gray-400 group-hover:text-white transition" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 group-hover:text-gray-300 transition">Scientist</h3>
              <p className="text-sm text-gray-400 mb-4 font-normal">
                Tests the most relevant models and delivers predictions through machine learning.
              </p>
            </div>

            {/* Analyst */}
            <div className="bg-black p-12 border border-gray-800 hover:border-gray-600 transition group">
              <div className="mb-6">
                <svg className="w-12 h-12 text-gray-400 group-hover:text-white transition" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 group-hover:text-gray-300 transition">Analyst</h3>
              <p className="text-sm text-gray-400 mb-4 font-normal">
                Interprets insights in real time with an LLM-powered AI assistant.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Data Visualization Section */}
      <section className="pt-0 pb-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-4xl font-extrabold mb-8">
                Why Small Businesses Need AutoML
              </h2>
              <div className="space-y-6">
                <p className="text-gray-400 font-normal leading-relaxed">
                  <span className="text-white font-semibold">Limited resources:</span> SMEs often lack the budget and in-house expertise to implement advanced AI projects. Traditional machine learning demands months of work and costly infrastructure. AutoML handles preparation, training and deployment on its own, lowering the barrier to entry.
                </p>
                <p className="text-gray-400 font-normal leading-relaxed">
                  <span className="text-white font-semibold">Innovation at speed:</span> In fast-moving markets, speed is a key differentiator. With no-code automation, companies can test new ideas, forecast demand, or validate marketing campaigns in days instead of months.
                </p>
                <p className="text-gray-400 font-normal leading-relaxed">
                  <span className="text-white font-semibold">Return on investment:</span> No-code tools consistently deliver value. By predicting churn, identifying profitable product sets, or uncovering hidden trends, AutoML enables companies to increase sales without adding extra costs.
                </p>
                <p className="text-gray-400 font-normal leading-relaxed">
                  <span className="text-white font-semibold">Empowered teams:</span> Marketing professionals, sales managers and operations teams can all benefit from AutoML. With drag-and-drop processes and natural-language interactions, they no longer need to rely on technical support; they can create and learn independently.
                </p>
              </div>
            </div>

            <div className="relative h-96 border border-gray-800 bg-gradient-to-br from-gray-900 to-black overflow-hidden">
              <div className="absolute inset-0">
                <svg className="w-full h-full" viewBox="0 0 400 400">
                  {/* Grid lines */}
                  <defs>
                    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(75, 85, 99, 0.3)" strokeWidth="0.5"/>
                    </pattern>
                  </defs>
                  <rect width="400" height="400" fill="url(#grid)" />

                  {/* Central node */}
                  <circle cx="200" cy="200" r="8" fill="white" opacity="0.8">
                    <animate attributeName="r" values="8;10;8" dur="2s" repeatCount="indefinite" />
                  </circle>

                  {/* Connecting lines and nodes */}
                  <line x1="200" y1="200" x2="100" y2="100" stroke="rgba(156, 163, 175, 0.4)" strokeWidth="1"/>
                  <circle cx="100" cy="100" r="5" fill="rgba(156, 163, 175, 0.6)"/>

                  <line x1="200" y1="200" x2="300" y2="100" stroke="rgba(156, 163, 175, 0.4)" strokeWidth="1"/>
                  <circle cx="300" cy="100" r="5" fill="rgba(156, 163, 175, 0.6)"/>

                  <line x1="200" y1="200" x2="100" y2="300" stroke="rgba(156, 163, 175, 0.4)" strokeWidth="1"/>
                  <circle cx="100" cy="300" r="5" fill="rgba(156, 163, 175, 0.6)"/>

                  <line x1="200" y1="200" x2="300" y2="300" stroke="rgba(156, 163, 175, 0.4)" strokeWidth="1"/>
                  <circle cx="300" cy="300" r="5" fill="rgba(156, 163, 175, 0.6)"/>

                  <line x1="200" y1="200" x2="50" y2="200" stroke="rgba(156, 163, 175, 0.4)" strokeWidth="1"/>
                  <circle cx="50" cy="200" r="5" fill="rgba(156, 163, 175, 0.6)"/>

                  <line x1="200" y1="200" x2="350" y2="200" stroke="rgba(156, 163, 175, 0.4)" strokeWidth="1"/>
                  <circle cx="350" cy="200" r="5" fill="rgba(156, 163, 175, 0.6)"/>

                  {/* Secondary connections */}
                  <line x1="100" y1="100" x2="300" y2="100" stroke="rgba(107, 114, 128, 0.2)" strokeWidth="0.5" strokeDasharray="2,2"/>
                  <line x1="100" y1="300" x2="300" y2="300" stroke="rgba(107, 114, 128, 0.2)" strokeWidth="0.5" strokeDasharray="2,2"/>
                  <line x1="100" y1="100" x2="100" y2="300" stroke="rgba(107, 114, 128, 0.2)" strokeWidth="0.5" strokeDasharray="2,2"/>
                  <line x1="300" y1="100" x2="300" y2="300" stroke="rgba(107, 114, 128, 0.2)" strokeWidth="0.5" strokeDasharray="2,2"/>

                  {/* Data points */}
                  <circle cx="150" cy="150" r="3" fill="rgba(209, 213, 219, 0.4)">
                    <animate attributeName="opacity" values="0.4;0.8;0.4" dur="3s" repeatCount="indefinite" />
                  </circle>
                  <circle cx="250" cy="150" r="3" fill="rgba(209, 213, 219, 0.4)">
                    <animate attributeName="opacity" values="0.4;0.8;0.4" dur="3.5s" repeatCount="indefinite" />
                  </circle>
                  <circle cx="150" cy="250" r="3" fill="rgba(209, 213, 219, 0.4)">
                    <animate attributeName="opacity" values="0.4;0.8;0.4" dur="2.5s" repeatCount="indefinite" />
                  </circle>
                  <circle cx="250" cy="250" r="3" fill="rgba(209, 213, 219, 0.4)">
                    <animate attributeName="opacity" values="0.4;0.8;0.4" dur="4s" repeatCount="indefinite" />
                  </circle>

                  {/* Orbital ring */}
                  <circle cx="200" cy="200" r="80" fill="none" stroke="rgba(75, 85, 99, 0.3)" strokeWidth="0.5" strokeDasharray="5,5">
                    <animateTransform attributeName="transform" type="rotate" from="0 200 200" to="360 200 200" dur="20s" repeatCount="indefinite"/>
                  </circle>
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* What Sets Axelle AI Apart Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-extrabold mb-16">What Sets Axelle AI Apart</h2>

          <div className="max-w-5xl mx-auto space-y-8">
            <div>
              <h3 className="text-xl font-bold text-white mb-3">Context-aware data quality checks and KPI identification:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                If your dataset is incomplete, Axelle AI steps in like a virtual consultant. After a few onboarding questions, it searches for the KPIs most relevant to your industry and suggests additional data to collect. Combined with automated cleaning and preprocessing, this ensures models are not only technically reliable but also business-aligned from the start.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-white mb-3">Smart model selection and transparency:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                Axelle AI's AutoML engine tests multiple algorithms, tunes hyper-parameters, and selects the model that best fits your objective. Models can be optimized for speed or accuracy, while technical users gain full access to the underlying code. This ensures trust and transparency for critical decisions.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-white mb-3">Conversational data analysis:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                An embedded chatbot lets you ask natural language questions such as "Which clients are most likely to cancel next month?" The system instantly provides visualizations and forecasts. Powered by large language models (LLMs), this capability brings advanced analytics within reach of non-technical users.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-white mb-3">Integrations and APIs:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                Axelle AI seamlessly integrates with spreadsheets, CSV files and CRMs. Models can be exported as REST APIs or embedded into web apps to automate processes such as personalized marketing or dynamic pricing.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-white mb-3">Easy-to-use interface:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                The dashboard features drag-and-drop elements, guided workflows and real-time metrics. It lets you track model performance and share dashboards, all in one place.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-extrabold mb-16">Use Cases</h2>

          <div className="max-w-5xl mx-auto space-y-8">
            <div>
              <h3 className="text-xl font-bold text-white mb-3">SME with heaps of data but no structure:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                The company lacks clarity on which fields truly matter. Axelle AI detects the most critical KPIs and builds a robust dataset.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-white mb-3">SME with a complete dataset but no data scientist:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                The company lacks the expertise. Axelle AI automatically tests models, deploys them, and delivers predictive insights.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-white mb-3">Mid-sized enterprise with data science department but high costs:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                The technical team faces rising expenses and dependence on overseas AutoML tools. Axelle AI offers a cost-efficient solution with full access to the underlying code.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-white mb-3">Solo entrepreneur with no technical expertise:</h3>
              <p className="text-gray-400 font-normal leading-relaxed">
                Wants to use data but doesn't know where to start. With Axelle AI's no-code workflows, it's as simple as answering a few questions, uploading data, and generating models.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* RGPD-compliant Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-extrabold mb-8">GDPR-Compliant</h2>

          <div className="max-w-4xl">
            <p className="text-gray-400 font-normal leading-relaxed">
              Based in Paris, Axelle AI is rooted in European values of data privacy, transparency and responsibility. Its vision is clear: to democratize AI so that any business can compete with larger businesses through data-driven decision making.
            </p>
          </div>
        </div>
      </section>

      {/* Founder Story & Mission Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-extrabold mb-8">Founder Story & Mission</h2>

          <div className="max-w-4xl">
            <p className="text-gray-400 font-normal leading-relaxed">
              Axelle AI was founded by Karthigan Vijey, a French engineer who had previously built a financial advisory business that generated €700,000 in revenue. After observing how many SMEs struggle to access advanced analytics, he designed a platform that non-experts can use without financial barriers, with support from a senior ML engineer who has worked on projects for BNP Paribas, Parrot, AWS and Palantir.
            </p>
          </div>
        </div>
      </section>

      {/* Getting Started Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-extrabold mb-8">Getting Started</h2>

          <div className="max-w-4xl">
            <p className="text-gray-400 font-normal leading-relaxed">
              Axelle AI is currently in an early pilot stage and preparing for launch in late 2025. <a href="#hero" className="text-white font-semibold hover:underline transition">Join&nbsp;waitlist&nbsp;now</a>
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-900 py-12 px-6 mt-20">
        <div className="max-w-7xl mx-auto text-center text-sm text-gray-600">
          <p>© 2025 Axelle AI</p>
        </div>
      </footer>
    </div>
  );
}
