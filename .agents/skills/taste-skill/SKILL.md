---
name: taste-skill
description: >-
  Enforces high-standard, authentic design taste in frontend code, UI layouts, and brand visual
  assets. Eliminates generic AI templates, centered low-contrast cards, and purple gradient clichés.
  Guides the agent through intentional typographic scales, harmonious palettes, deliberate spatial
  pacing, and senior frontend craftsmanship. Triggers: 'frontend design', 'make it look good',
  'improve UI', 'high taste', 'design system', 'styling'.
---

# Taste Skill: High-Standard Frontend & Visual Craftsmanship

Taste is the deliberate rejection of default, unconsidered choices. AI agents notoriously produce "AI-slop UI" (centered cards with purple-to-blue gradients, low-contrast text, uniform border-radius on everything, and lack of visual hierarchy). This skill provides concrete rules to enforce senior-level aesthetic taste.

---

## 1. The Anti-Slop Design Principles

### Rule 1: Kill the "AI Card" Default
- **Never** default to a floating centered card on a dark gradient background with a glowing purple outline.
- Use intentional layout composition: asymmetric grids, editorial sidebars, split-screen hierarchies, or full-bleed containers with hairline dividers (`1px solid var(--border)`).

### Rule 2: Typographic Discipline & Scale
- Establish a strict 3-tier hierarchy:
  1. **Display / Headlines:** Heavyweight, tight tracking (`letter-spacing: -0.02em` to `-0.04em`), intentional line-height (`1.05` to `1.2`).
  2. **Body:** High-legibility sans-serif (Inter, Geist, SF Pro, or system font stack) with generous line-height (`1.5` to `1.6`) and neutral contrast (`#E2E8F0` or `#334155`).
  3. **Labels / Metadata:** Monospace or clean micro-caps (`font-size: 11px`, `letter-spacing: 0.05em`, `text-transform: uppercase`, `font-weight: 600`).
- **Never** use more than two typeface families in a single project (one display/body, one monospace for code/telemetry).

### Rule 3: The 3-Color Maximum Palette
- Limit palettes to:
  - **Base Canvas:** Deep inky dark (`#030712`, `#090D16`) or crisp porcelain light (`#FFFFFF`, `#F8FAFC`).
  - **Structural Borders:** Subtle slate/zinc hairline (`#1E293B`, `#E2E8F0`).
  - **Single Signature Accent:** One electric signal color (e.g., cobalt `#2563EB`, emerald `#10B981`, or signal cyan `#0284C7`). Do not paint a rainbow.

### Rule 4: Micro-Interactions & State Transition
- Hover states must feel physical and responsive:
  - Fast transitions: `transition: all 120ms cubic-bezier(0.16, 1, 0.3, 1)`.
  - Subtle elevation or border-color shifts rather than dramatic scaling.
  - Active tap feedback: `transform: scale(0.98)`.

---

## 2. Component Design Checklist

Before delivering any frontend component:
- [ ] Is there an obvious visual anchor (one thing draws the eye first)?
- [ ] Are paddings and margins rhythmic (multiples of 4px / 8px / 16px / 24px)?
- [ ] Does every interactive element have clear `:hover`, `:focus-visible`, and `:disabled` states?
- [ ] Is text contrast strictly WCAG AA compliant (minimum 4.5:1 ratio for normal text)?
- [ ] Are empty states and loading skeletons designed with the same care as populated states?
