---
name: awesome-design-md
description: >-
  Standardized DESIGN.md system for communicating design systems, tokens, and aesthetic constraints
  to AI agents. Curates battle-tested design specifications (Vercel, Linear, Stripe, Apple, Tailwind).
  Use to create, validate, and apply DESIGN.md files in a repository to maintain visual consistency.
---

# Awesome DESIGN.md: Design Systems for AI Agents

`DESIGN.md` is the plain-text standard for design systems in the era of agentic software development. It bridges the gap between Figma design tokens and AI code generation.

## 1. The DESIGN.md Specification

A standard `DESIGN.md` file consists of two parts:
1. **YAML Frontmatter (Machine-Readable Tokens):** Color scales, font families, radius tokens, breakpoints, and spacing units.
2. **Markdown Body (Human/AI Guidelines):** Design philosophy, component composition rules, tone of voice, and accessibility constraints.

### Canonical Template:

```markdown
---
name: "Modern Engineering Dark"
version: "1.0.0"
tokens:
  colors:
    canvas: "#030712"
    surface: "#0B0F19"
    surface-raised: "#111827"
    border: "#1F2937"
    border-subtle: "#111827"
    text-primary: "#F8FAFC"
    text-secondary: "#94A3B8"
    text-muted: "#64748B"
    accent-primary: "#0284C7"
    accent-hover: "#0369A1"
    success: "#10B981"
    warning: "#F59E0B"
    destructive: "#EF4444"
  typography:
    font-sans: "-apple-system, BlinkMacSystemFont, 'Inter', 'Geist', sans-serif"
    font-mono: "'JetBrains Mono', 'Geist Mono', 'SF Mono', monospace"
    scale:
      xs: "11px"
      sm: "13px"
      base: "15px"
      lg: "18px"
      xl: "24px"
      display: "36px"
  radii:
    none: "0px"
    sm: "4px"
    md: "8px"
    lg: "12px"
    full: "9999px"
---

# Design System Guidelines

## 1. Philosophy: Product-First Precision
- Every screen should feel fast, responsive, and distraction-free.
- Favor hairline borders (`1px solid var(--border)`) over dramatic drop shadows.

## 2. Spacing Grid
- Always use the 4px baseline grid (4, 8, 12, 16, 24, 32, 48, 64px).
```

## 2. Workflows with DESIGN.md
1. **Bootstrap Project Aesthetic:** Place `DESIGN.md` in the root of the workspace.
2. **Sync to CSS Variables:** Convert YAML tokens into `:root { --color-canvas: ... }` in `index.css`.
3. **Audit Components:** Review all PRs and UI additions against the active `DESIGN.md`.
