import React from "react";
import { EN, formatReadableDate } from "../i18n/en";
import type { HealthResponse } from "../types/api";

interface AboutLimitationsSectionProps {
  health?: HealthResponse | null;
}

export const AboutLimitationsSection: React.FC<AboutLimitationsSectionProps> = ({ health }) => {
  const backendKey = health?.weather_backend || health?.db_role || "bundled_demo_db";
  const dbRoleKey = health?.db_role || backendKey;
  const resolvedModeKey =
    backendKey === "turso_replica" || backendKey === "turso_remote"
      ? backendKey
      : dbRoleKey === "bundled_demo_db"
        ? "bundled_demo_db"
        : backendKey === "sample_csv"
          ? "sample_csv"
          : "local_or_production_db";
  const modeLabel = EN.app.snapshotModes[resolvedModeKey] ?? EN.app.snapshotModes.local_or_production_db;
  const scanStr = health?.scan_window_end ? formatReadableDate(health.scan_window_end) : null;

  return (
    <section className="panel" aria-label={EN.governance.heading}>
      <h2>{EN.governance.heading}</h2>
      <p className="panel-subtitle">{EN.governance.subtitle}</p>
      <div className="governance-grid">
        {EN.governance.cards.map((card, idx) => (
          <article key={card.title} className="governance-card">
            <h3>{card.title}</h3>
            <p>{card.body}</p>
            {idx === EN.governance.cards.length - 1 && (
              <p className="snapshot-mode-status" data-testid="dynamic-snapshot-mode">
                {EN.governance.snapshotDynamicLine(modeLabel, dbRoleKey, scanStr)}
              </p>
            )}
            {"links" in card && Array.isArray(card.links) && card.links.length > 0 && (
              <ul className="citation-list">
                {card.links.map((lnk) => (
                  <li key={lnk.url}>
                    <a href={lnk.url} target="_blank" rel="noopener noreferrer">
                      {lnk.label}
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </article>
        ))}
      </div>
    </section>
  );
};
