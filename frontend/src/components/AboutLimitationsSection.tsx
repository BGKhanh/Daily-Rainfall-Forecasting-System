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
          </article>
        ))}
      </div>
    </section>
  );
};
