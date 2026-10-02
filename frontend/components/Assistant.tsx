"use client";

import { useEffect, useState } from "react";
import { api, ApiError, type AssistantAnswer } from "@/lib/api";

export default function Assistant({ onFocusPlace }: { onFocusPlace: (placeId: string) => void }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AssistantAnswer | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [planner, setPlanner] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .suggestions()
      .then((data) => {
        setSuggestions(data.suggestions);
        setPlanner(data.planner);
      })
      .catch(() => setSuggestions([]));
  }, []);

  const ask = async (text: string) => {
    if (!text.trim() || busy) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(await api.ask(text));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The assistant is unavailable.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="section">
        <h2>Ask a question</h2>
        <form
          className="ask-row"
          onSubmit={(event) => {
            event.preventDefault();
            void ask(question);
          }}
        >
          <input
            aria-label="Question about satellite observations"
            minLength={3}
            maxLength={500}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Which reservoirs are losing water?"
            disabled={busy}
          />
          <button className="primary" type="submit" disabled={busy || question.trim().length < 3}>
            {busy ? <span className="spinner" /> : "Ask"}
          </button>
        </form>
        <p className="hint">
          {planner === "claude"
            ? "Claude routes your question to an analysis module; the numbers are computed from imagery, never recalled."
            : "Keyword routing. Set ANTHROPIC_API_KEY for free-form question planning."}
        </p>
      </div>

      {busy && (
        <div className="section">
          <div className="loading">
            <span className="spinner" /> Downloading and analysing imagery…
          </div>
          <p className="hint">
            Scans read several areas from the satellite archive. Expect 10–40 seconds.
          </p>
        </div>
      )}

      {error && (
        <div className="section">
          <p className="error">{error}</p>
        </div>
      )}

      {answer && !busy && (
        <>
          <div className="section">
            <h2>Answer</h2>
            <p className="summary" style={{ whiteSpace: "pre-line" }}>
              {answer.answer}
            </p>
            {answer.plan && (
              <p className="hint" style={{ marginTop: 8 }}>
                Ran <strong>{answer.plan.module}</strong> over {answer.plan.window.start} →{" "}
                {answer.plan.window.end}
                {answer.plan.baseline
                  ? `, against ${answer.plan.baseline.start} → ${answer.plan.baseline.end}`
                  : ""}
                . Planned by {answer.plan.planner}.
              </p>
            )}
          </div>

          {answer.results && answer.results.length > 1 && (
            <div className="section">
              <h2>Ranked areas</h2>
              {answer.results.map((entry) => (
                <button
                  className="rank"
                  key={entry.place.id}
                  style={{ display: "block", width: "100%", textAlign: "left" }}
                  onClick={() => onFocusPlace(entry.place.id)}
                >
                  <span className="rank-head">
                    <span className="rank-num">#{entry.rank ?? "—"}</span>
                    <span className="rank-name">{entry.place.name}</span>
                  </span>
                  <span className="rank-body">{entry.summary}</span>
                </button>
              ))}
            </div>
          )}

          {answer.gaps && answer.gaps.length > 0 && (
            <div className="section">
              <h2>No data</h2>
              <p className="caveat">
                {answer.gaps.length} area(s) returned no usable imagery and are absent from the
                ranking — that is a gap in observation, not evidence they are unaffected.
              </p>
              {answer.gaps.map((gap) => (
                <span className="chip" key={gap.place.id}>
                  {gap.place.name}
                </span>
              ))}
            </div>
          )}

          {answer.suggestions && (
            <div className="section">
              <h2>Try instead</h2>
              {answer.suggestions.map((text) => (
                <button className="suggestion" key={text} onClick={() => void ask(text)}>
                  {text}
                </button>
              ))}
            </div>
          )}
        </>
      )}

      {!answer && !busy && suggestions.length > 0 && (
        <div className="section">
          <h2>Examples</h2>
          {suggestions.map((text) => (
            <button
              className="suggestion"
              key={text}
              onClick={() => {
                setQuestion(text);
                void ask(text);
              }}
            >
              {text}
            </button>
          ))}
        </div>
      )}
    </>
  );
}
