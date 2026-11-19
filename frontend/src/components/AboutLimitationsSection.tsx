import React from "react";
import { EN } from "../i18n/en";

export const AboutLimitationsSection: React.FC = () => {
  return (
    <section className="panel" aria-label={EN.governance.heading}>
      <h2>{EN.governance.heading}</h2>
      <p className="panel-subtitle">{EN.governance.subtitle}</p>
      <div className="governance-grid">
        {EN.governance.cards.map((card) => (
          <article key={card.title} className="governance-card">
            <h3>{card.title}</h3>
            <p>{card.body}</p>
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
