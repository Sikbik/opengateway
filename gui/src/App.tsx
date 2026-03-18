import powerOrbOffline from "./assets/power-orb-offline.svg";
import powerOrbOnline from "./assets/power-orb-online.svg";
import { createPortal } from "react-dom";
import {
  startTransition,
  useEffect,
  useDeferredValue,
  useRef,
  useState,
} from "react";
import { call, runtimeMode } from "./runtime";
import type {
  AcpGuiSnapshot,
  AcpGuiSessionDetail,
  AppSnapshot,
  CommandResult,
  DroidRecord,
  HealthState,
} from "./types";

type TabKey = "dashboard" | "droids" | "factory" | "acp" | "logs";
type DroidFilterKey = "all" | "custom" | "inherit" | "issues";
type DroidSortMode = "name" | "scope" | "custom" | "flagged";
type DroidPresetKey = "none" | "reviewers" | "workers" | "inherited";
type SelectOption = { value: string; label: string };
type MenuStyle = {
  left: number;
  width: number;
  maxHeight: number;
} & (
  | { top: number; bottom?: never }
  | { bottom: number; top?: never }
);

const TABS: Array<{ key: TabKey; label: string; index: string }> = [
  { key: "dashboard", label: "Overview", index: "01" },
  { key: "droids", label: "Droids", index: "02" },
  { key: "factory", label: "Factory", index: "03" },
  { key: "acp", label: "ACP", index: "04" },
  { key: "logs", label: "Logs", index: "05" },
];

const POLL_INTERVAL_MS = 5000;
const DROID_FILTERS: Array<{ key: DroidFilterKey; label: string }> = [
  { key: "all", label: "All" },
  { key: "custom", label: "Custom" },
  { key: "inherit", label: "Inherited" },
  { key: "issues", label: "Flagged" },
];
const DROID_SORTS: Array<{ key: DroidSortMode; label: string }> = [
  { key: "name", label: "Name" },
  { key: "scope", label: "Scope" },
  { key: "custom", label: "Custom first" },
  { key: "flagged", label: "Flagged first" },
];
const DROID_PRESETS: Array<{ key: DroidPresetKey; label: string }> = [
  { key: "none", label: "All roles" },
  { key: "reviewers", label: "Reviewers" },
  { key: "workers", label: "Workers" },
  { key: "inherited", label: "Inherited only" },
];

const HEALTH_COPY: Record<HealthState, string> = {
  online: "Nominal",
  degraded: "Drifting",
  offline: "Offline",
};

function formatMinutes(minutes: number | null) {
  if (minutes == null) {
    return "Unknown";
  }
  if (minutes >= 1440) {
    const days = Math.floor(minutes / 1440);
    const hours = Math.floor((minutes % 1440) / 60);
    return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
  }
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const remainder = minutes % 60;
    return remainder > 0 ? `${hours}h ${remainder}m` : `${hours}h`;
  }
  return `${minutes}m`;
}

function formatModel(
  value: string | null | undefined,
  fallback = "Unassigned",
) {
  if (!value) {
    return fallback;
  }
  if (value === "inherit") {
    return "Inherit Factory defaults";
  }
  return value.replace(/^custom:/, "");
}

function compactValue(value: string | null | undefined, keep = 10) {
  if (!value) {
    return "Unavailable";
  }
  if (value.length <= keep * 2 + 3) {
    return value;
  }
  return `${value.slice(0, keep)}...${value.slice(-keep)}`;
}

function formatUpdatedAt(timestamp: number | null | undefined) {
  if (!timestamp) {
    return "No live snapshot";
  }
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatLogTimestamp(timestamp: number | null) {
  if (!timestamp) {
    return "Legacy";
  }
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatSessionTimestamp(timestamp: number | null | undefined) {
  if (!timestamp) {
    return "Never";
  }
  return new Date(timestamp).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatSessionState(value: string | null | undefined) {
  if (!value) {
    return "Unknown";
  }
  return value.replace(/[_-]+/g, " ");
}

function formatSessionEvent(value: string | null | undefined) {
  if (!value) {
    return "No activity";
  }
  return value.replace(/[._-]+/g, " ");
}

function getAgentTone(agent: {
  ready: boolean;
  issue: string | null;
}): HealthState {
  if (agent.ready) {
    return "online";
  }
  if (agent.issue) {
    return "offline";
  }
  return "degraded";
}

function parseLogLine(line: string) {
  const match = line.match(/^\[(\d{13})\]\s+(.*)$/);
  if (!match) {
    return {
      raw: line,
      message: line,
      timestamp: null as number | null,
    };
  }

  return {
    raw: line,
    message: match[2],
    timestamp: Number(match[1]),
  };
}

function SensitiveValue({
  value,
  keep = 16,
  fallback = "Unavailable",
}: {
  value: string | null | undefined;
  keep?: number;
  fallback?: string;
}) {
  if (!value) {
    return <span>{fallback}</span>;
  }

  return (
    <span className="sensitive-inline" tabIndex={0}>
      {compactValue(value, keep)}
    </span>
  );
}

function HelpTip({
  text,
  label = "More info",
  placement = "top",
  align = "center",
}: {
  text: string;
  label?: string;
  placement?: "top" | "bottom";
  align?: "start" | "center" | "end";
}) {
  return (
    <span
      className={`help-tip help-tip--${placement} help-tip--${align}`}
      tabIndex={0}
      aria-label={`${label}: ${text}`}
    >
      <span className="help-tip__dot">?</span>
      <span className="help-tip__bubble" role="tooltip">
        {text}
      </span>
    </span>
  );
}

function ThemedSelect({
  value,
  options,
  onChange,
  placeholder = "Select option",
  disabled = false,
  className = "",
}: {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const selectedOption = options.find((option) => option.value === value);
  const [menuStyle, setMenuStyle] = useState<MenuStyle | null>(null);
  const [menuDirection, setMenuDirection] = useState<"up" | "down">("down");

  useEffect(() => {
    if (!open) {
      return;
    }

    function updateMenuPosition() {
      const trigger = triggerRef.current;
      if (!trigger) {
        return;
      }

      const rect = trigger.getBoundingClientRect();
      const gutter = 12;
      const gap = 8;
      const roomBelow = window.innerHeight - rect.bottom - gutter;
      const roomAbove = rect.top - gutter;
      const openUp = roomBelow < 220 && roomAbove > roomBelow;
      const availableHeight = openUp ? roomAbove : roomBelow;
      const maxHeight = Math.max(140, Math.min(320, availableHeight - gap));
      const width = rect.width;
      const left = Math.max(
        gutter,
        Math.min(rect.left, window.innerWidth - width - gutter),
      );

      if (openUp) {
        setMenuDirection("up");
        setMenuStyle({
          left,
          width,
          maxHeight,
          bottom: window.innerHeight - rect.top + gap,
        });
        return;
      }

      setMenuDirection("down");
      setMenuStyle({
        left,
        width,
        maxHeight,
        top: rect.bottom + gap,
      });
    }

    function handlePointerDown(event: PointerEvent) {
      const target = event.target as Node | null;
      if (rootRef.current?.contains(target) || menuRef.current?.contains(target)) {
        return;
      }
      setOpen(false);
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    updateMenuPosition();
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    window.addEventListener("resize", updateMenuPosition);
    window.addEventListener("scroll", updateMenuPosition, true);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", updateMenuPosition);
      window.removeEventListener("scroll", updateMenuPosition, true);
    };
  }, [open]);

  return (
    <div
      ref={rootRef}
      className={`themed-select${open ? " themed-select--open" : ""}${
        className ? ` ${className}` : ""
      }`}
    >
      <button
        ref={triggerRef}
        type="button"
        className="themed-select__trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
      >
        <span
          className={
            selectedOption ? "themed-select__value" : "themed-select__placeholder"
          }
        >
          {selectedOption?.label ?? placeholder}
        </span>
        <span className="themed-select__chevron" aria-hidden="true">
          ▾
        </span>
      </button>
      {open && menuStyle
        ? createPortal(
            <div
              ref={menuRef}
              className={`themed-select__menu themed-select__menu--${menuDirection}`}
              role="listbox"
              style={menuStyle}
            >
              {options.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  role="option"
                  aria-selected={option.value === value}
                  className={
                    option.value === value
                      ? "themed-select__option themed-select__option--active"
                      : "themed-select__option"
                  }
                  onClick={() => {
                    onChange(option.value);
                    setOpen(false);
                  }}
                >
                  {option.label}
                </button>
              ))}
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}

function App() {
  const [snapshot, setSnapshot] = useState<AppSnapshot | null>(null);
  const [acpSnapshot, setAcpSnapshot] = useState<AcpGuiSnapshot | null>(null);
  const [acpSessionDetail, setAcpSessionDetail] =
    useState<AcpGuiSessionDetail | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [doctorOutput, setDoctorOutput] = useState("");
  const [activeTab, setActiveTab] = useState<TabKey>("dashboard");
  const [selectedModels, setSelectedModels] = useState<Record<string, string>>(
    {},
  );
  const [bulkModel, setBulkModel] = useState("");
  const [bulkResult, setBulkResult] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [droidFilter, setDroidFilter] = useState<DroidFilterKey>("all");
  const [sortMode, setSortMode] = useState<DroidSortMode>("name");
  const [activePreset, setActivePreset] = useState<DroidPresetKey>("none");
  const [compactRows, setCompactRows] = useState(true);
  const [collapsedScopes, setCollapsedScopes] = useState<
    Record<"workspace" | "machine", boolean>
  >({
    workspace: false,
    machine: false,
  });
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [acpError, setAcpError] = useState<string | null>(null);
  const [acpDetailError, setAcpDetailError] = useState<string | null>(null);
  const [acpDetailNotice, setAcpDetailNotice] = useState<string | null>(null);
  const [acpDetailLimit, setAcpDetailLimit] = useState(12);
  const [selectedAcpSessionId, setSelectedAcpSessionId] = useState<string | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);
  const mode = runtimeMode();
  const deferredSearchQuery = useDeferredValue(searchQuery);

  async function refreshAll() {
    try {
      const nextSnapshot = await call<AppSnapshot>("load_snapshot");
      const nextLogs = await call<string[]>("tail_logs", { limit: 160 });

      if (!mounted.current) {
        return;
      }

      startTransition(() => {
        setSnapshot(nextSnapshot);
        setLogs(nextLogs);
        setSelectedModels((current) => {
          const draft = { ...current };
          for (const droid of nextSnapshot.droids) {
            if (!draft[droid.path] && droid.model) {
              draft[droid.path] = droid.model;
            }
          }
          return draft;
        });
      });
      if (!bulkModel) {
        setBulkModel(
          nextSnapshot.models.find(
            (option) => option.model === "custom:gpt-5.4(xhigh)",
          )?.model ??
            nextSnapshot.models[0]?.model ??
            "inherit",
        );
      }
      setError(null);
    } catch (cause) {
      if (!mounted.current) {
        return;
      }
      setError(String(cause));
    }
  }

  async function refreshAcp() {
    try {
      const nextSnapshot = await call<AcpGuiSnapshot>("load_acp_snapshot");
      if (!mounted.current) {
        return;
      }
      const nextSessionId = nextSnapshot.sessions.some(
        (session) => session.sessionId === selectedAcpSessionId,
      )
        ? selectedAcpSessionId
        : nextSnapshot.sessions[0]?.sessionId ?? null;
      startTransition(() => {
        setAcpSnapshot(nextSnapshot);
        setSelectedAcpSessionId(nextSessionId);
      });
      setAcpError(null);
      if (nextSessionId) {
        await refreshAcpSessionDetail(nextSessionId, acpDetailLimit);
      } else if (mounted.current) {
        startTransition(() => {
          setAcpSessionDetail(null);
        });
        setAcpDetailError(null);
      }
    } catch (cause) {
      if (!mounted.current) {
        return;
      }
      setAcpError(String(cause));
    }
  }

  async function refreshAcpSessionDetail(sessionId: string, limit = acpDetailLimit) {
    try {
      const nextDetail = await call<AcpGuiSessionDetail>("load_acp_session_detail", {
        sessionId,
        limit,
      });
      if (!mounted.current) {
        return;
      }
      startTransition(() => {
        setAcpSessionDetail(nextDetail);
      });
      setAcpDetailError(null);
    } catch (cause) {
      if (!mounted.current) {
        return;
      }
      setAcpDetailError(String(cause));
    }
  }

  useEffect(() => {
    mounted.current = true;
    void refreshAll();
    const timer = window.setInterval(() => {
      void refreshAll();
    }, POLL_INTERVAL_MS);

    return () => {
      mounted.current = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (activeTab !== "acp") {
      return;
    }

    void refreshAcp();
    const timer = window.setInterval(() => {
      void refreshAcp();
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [activeTab, selectedAcpSessionId, acpDetailLimit]);

  async function copyAcpPath(label: string, value: string) {
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error("Clipboard write is unavailable in this runtime.");
      }
      await navigator.clipboard.writeText(value);
      setAcpDetailNotice(`${label} copied.`);
      setAcpDetailError(null);
    } catch (cause) {
      setAcpDetailError(String(cause));
    }
  }

  async function runAction(
    action: string,
    task: () => Promise<unknown>,
    onSuccess?: (value: unknown) => void,
  ): Promise<boolean> {
    setBusyAction(action);
    setError(null);
    let shouldRefresh = false;
    try {
      const result = await task();
      onSuccess?.(result);
      shouldRefresh = true;
      return true;
    } catch (cause) {
      setError(String(cause));
      return false;
    } finally {
      if (mounted.current) {
        setBusyAction(null);
      }
      if (shouldRefresh) {
        void refreshAll();
      }
    }
  }

  async function handleDoctor() {
    await runAction(
      "doctor",
      () => call<CommandResult>("run_doctor"),
      (result) => setDoctorOutput((result as CommandResult).output.trim()),
    );
  }

  async function handlePowerToggle() {
    if (!snapshot) {
      return;
    }

    if (snapshot.gateway.running) {
      await runAction("stop", () => call<CommandResult>("stop_gateway"));
    } else {
      await runAction("start", () => call<CommandResult>("start_gateway"));
    }
  }

  async function handleModelSave(droid: DroidRecord) {
    const model = selectedModels[droid.path];
    if (!model || model === droid.model) {
      return;
    }

    const saved = await runAction(`save:${droid.path}`, () =>
      call<DroidRecord>("set_droid_model", { path: droid.path, model }),
    );
    if (saved) {
      await runAction("sync:auto", () => call<CommandResult>("sync_factory"));
    }
  }

  async function handleBulkAssign(
    action: string,
    predicate: (droid: DroidRecord) => boolean,
    label: string,
  ) {
    if (!snapshot || !bulkModel) {
      return;
    }

    const targets = snapshot.droids.filter(predicate);
    if (targets.length === 0) {
      setBulkResult(`No droids matched ${label.toLowerCase()}.`);
      return;
    }

    const applied = await runAction(
      `bulk:${action}`,
      async () => {
        for (const droid of targets) {
          await call<DroidRecord>("set_droid_model", {
            path: droid.path,
            model: bulkModel,
          });
        }
        return targets.length;
      },
      (count) => {
        const nextModelLabel =
          modelOptions.find((option) => option.model === bulkModel)
            ?.displayName ?? formatModel(bulkModel);
        setSelectedModels((current) => {
          const draft = { ...current };
          for (const droid of targets) {
            draft[droid.path] = bulkModel;
          }
          return draft;
        });
        setBulkResult(
          `Applied ${nextModelLabel} to ${count as number} ${label.toLowerCase()}.`,
        );
      },
    );
    if (applied) {
      await runAction("sync:auto", () => call<CommandResult>("sync_factory"));
    }
  }

  function applyPreset(preset: DroidPresetKey) {
    setActivePreset(preset);
    setSearchQuery("");
    setDroidFilter(preset === "inherited" ? "inherit" : "all");
  }

  const modelOptions = snapshot?.models ?? [];
  const droids = snapshot?.droids ?? [];
  const searchNeedle = deferredSearchQuery.trim().toLowerCase();
  const matchesPreset = (droid: DroidRecord) => {
    switch (activePreset) {
      case "reviewers":
        return /(review|auditor|architect|validator)/i.test(droid.name);
      case "workers":
        return /(worker|probe)/i.test(droid.name);
      case "inherited":
        return droid.kind === "inherit";
      default:
        return true;
    }
  };
  const matchesFilter = (droid: DroidRecord) => {
    switch (droidFilter) {
      case "custom":
        return droid.kind === "custom";
      case "inherit":
        return droid.kind === "inherit";
      case "issues":
        return droid.issues.length > 0;
      default:
        return true;
    }
  };
  const matchesSearch = (droid: DroidRecord) => {
    if (!searchNeedle) {
      return true;
    }

    return [droid.name, droid.path, droid.model ?? "", formatModel(droid.model)]
      .join(" ")
      .toLowerCase()
      .includes(searchNeedle);
  };
  const compareDroids = (left: DroidRecord, right: DroidRecord) => {
    const nameOrder = left.name.localeCompare(right.name);
    const scopeOrder =
      (left.scope === right.scope ? 0 : left.scope === "workspace" ? -1 : 1) ||
      nameOrder;
    const customRank = (droid: DroidRecord) => {
      switch (droid.kind) {
        case "custom":
          return 0;
        case "inherit":
          return 1;
        case "builtin":
          return 2;
        default:
          return 3;
      }
    };
    const flaggedOrder =
      (right.issues.length > 0 ? 1 : 0) - (left.issues.length > 0 ? 1 : 0);
    const customOrder = customRank(left) - customRank(right);

    switch (sortMode) {
      case "scope":
        return scopeOrder;
      case "custom":
        return customOrder || nameOrder;
      case "flagged":
        return flaggedOrder || customOrder || nameOrder;
      default:
        return nameOrder;
    }
  };
  const filteredDroids = droids
    .filter(
      (droid) =>
        matchesPreset(droid) && matchesFilter(droid) && matchesSearch(droid),
    )
    .sort(compareDroids);
  const groupedAllDroids = droids.reduce(
    (groups, droid) => {
      groups[droid.scope].push(droid);
      return groups;
    },
    { workspace: [] as DroidRecord[], machine: [] as DroidRecord[] },
  );
  const groupedDroids = filteredDroids.reduce(
    (groups, droid) => {
      groups[droid.scope].push(droid);
      return groups;
    },
    { workspace: [] as DroidRecord[], machine: [] as DroidRecord[] },
  );
  const filterCounts = {
    all: droids.length,
    custom: droids.filter((droid) => droid.kind === "custom").length,
    inherit: droids.filter((droid) => droid.kind === "inherit").length,
    issues: droids.filter((droid) => droid.issues.length > 0).length,
  };
  const customDroidCount = filterCounts.custom;
  const inheritedDroidCount = filterCounts.inherit;
  const issueCount = filterCounts.issues;
  const sortOptions = DROID_SORTS.map((sort) => ({
    value: sort.key,
    label: sort.label,
  }));
  const modelChoices = modelOptions.map((option) => ({
    value: option.model,
    label: option.displayName,
  }));
  const bulkModelChoices = [
    { value: "inherit", label: "Inherit Factory defaults" },
    ...modelChoices,
  ];

  const missionCards = snapshot
    ? [
        {
          label: "Orchestrator",
          value: snapshot.factory.missionModels.orchestrator ?? "unset",
        },
        {
          label: "Worker",
          value: snapshot.factory.missionModels.worker ?? "unset",
        },
        {
          label: "Validator",
          value: snapshot.factory.missionModels.validationWorker ?? "unset",
        },
      ]
    : [];

  const heroChips = snapshot
    ? [
        { label: "runtime", value: mode },
        { label: "health", value: HEALTH_COPY[snapshot.gateway.health] },
        { label: "droids", value: String(snapshot.droids.length) },
      ]
    : [];

  const gatewayStageMetrics = snapshot
    ? [
        {
          label: "Active model",
          value: formatModel(snapshot.gateway.preferredModel),
        },
        {
          label: "Session default",
          value: formatModel(snapshot.factory.sessionDefaultModel),
        },
        {
          label: "Custom models",
          value: String(snapshot.factory.settingsCustomModelCount),
        },
        {
          label: "Pinned droids",
          value: `${customDroidCount}/${snapshot.droids.length}`,
        },
      ]
    : [];

  const powerOrb = snapshot?.gateway.running ? powerOrbOnline : powerOrbOffline;
  const powerLabel = snapshot?.gateway.running
    ? "Online"
    : busyAction === "start"
      ? "Engaging"
      : busyAction === "stop"
        ? "Cooling Down"
        : "Offline";
  const logEntries = logs.slice().reverse().map(parseLogLine);
  const acpAgents = acpSnapshot?.agents ?? [];
  const acpIssues = acpSnapshot?.issues ?? [];
  const acpSessions = acpSnapshot?.sessions ?? [];
  const selectedAcpSession =
    acpSessions.find((session) => session.sessionId === selectedAcpSessionId) ?? null;
  const readyAgentCount = acpAgents.filter((agent) => agent.ready).length;
  const acpHealthTone: HealthState = acpSnapshot
    ? readyAgentCount > 0
      ? "online"
      : "offline"
    : "degraded";
  const acpHealthLabel = acpSnapshot
    ? readyAgentCount > 0
      ? "Ready"
      : "Not ready"
    : "Loading";
  const acpMetrics = acpSnapshot
    ? [
        {
          label: "Adapters ready",
          value: `${readyAgentCount}/${acpAgents.length}`,
        },
        {
          label: "Recorded sessions",
          value: String(acpSessions.length),
        },
        {
          label: "Prompt completions",
          value: String(acpSnapshot.metrics.promptsCompleted),
        },
        {
          label: "Runtime failures",
          value: String(acpSnapshot.metrics.runtimeFailures),
        },
        {
          label: "Open issues",
          value: String(acpIssues.length),
        },
      ]
    : [];
  const acpFacts = acpSnapshot
    ? [
        {
          label: "Command",
          value: acpSnapshot.command,
        },
        {
          label: "Transport",
          value: acpSnapshot.transport,
        },
        {
          label: "Process model",
          value: acpSnapshot.processModel,
        },
        {
          label: "Profile",
          value: acpSnapshot.experimental ? "Experimental" : "Stable",
        },
      ]
    : [];

  return (
    <div className="shell">
      <div className="shell__aurora" />
      <div className="shell__noise" />

      <header className="hero panel">
        <div className="hero__copy">
          <div className="hero__eyebrow">
            <span className="eyebrow">Factory Control</span>
            <span className="status-led status-led--live" />
            <span className="hero__timestamp">
              Snapshot {formatUpdatedAt(snapshot?.generatedAt)}
            </span>
          </div>

          <h1>
            Command the gateway.
            <span>Retarget every Droid from one flight deck.</span>
          </h1>

          <p className="lede">
            Operate OpenGateway, inspect Factory defaults, and pin custom-model
            routing without bouncing between settings files and subagent
            markdown.
          </p>

          <div className="hero__chips">
            {heroChips.map((chip) => (
              <div key={chip.label} className="hero-chip">
                <span>{chip.label}</span>
                <strong>{chip.value}</strong>
              </div>
            ))}
          </div>

          <div className="hero__statusbar">
            <span
              className={`health-pill health-pill--${snapshot?.gateway.health ?? "offline"}`}
            >
              {HEALTH_COPY[snapshot?.gateway.health ?? "offline"]}
            </span>
            {snapshot?.workspacePath ? (
              <span className="hero__path">
                Workspace {compactValue(snapshot.workspacePath, 22)}
              </span>
            ) : snapshot?.factory.homePath ? (
              <span className="hero__path">
                Factory {compactValue(snapshot.factory.homePath, 22)}
              </span>
            ) : null}
            {snapshot?.environment.isWsl ? (
              <span className="hero__runtime-note">
                Browser bridge mode active for WSL.
              </span>
            ) : null}
          </div>
        </div>

        <nav className="tabs tabs--embedded" aria-label="Sections">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              className={tab.key === activeTab ? "tab tab--active" : "tab"}
              onClick={() => setActiveTab(tab.key)}
            >
              <span>{tab.index}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      {error ? <div className="error-banner panel">{error}</div> : null}

      <main className="content">
        <section
          className={activeTab === "dashboard" ? "view view--active" : "view"}
        >
          <div className="overview-grid">
            <article className="panel gateway-stage">
              <div className="gateway-stage__top">
                <div className="gateway-stage__copy">
                  <p className="eyebrow">Gateway Core</p>
                  <div className="gateway-stage__headline">
                    <div>
                      <h2>
                        {snapshot?.gateway.running
                          ? "Gateway live"
                          : "Gateway idle"}
                      </h2>
                      <p>
                        {snapshot?.gateway.apiBaseUrl ??
                          "http://127.0.0.1:42069/v1"}
                      </p>
                    </div>
                    <span
                      className={`health-pill health-pill--${snapshot?.gateway.health ?? "offline"}`}
                    >
                      {HEALTH_COPY[snapshot?.gateway.health ?? "offline"]}
                    </span>
                  </div>
                  <div className="gateway-stage__meta">
                    <div>
                      <span>PID</span>
                      <strong>{snapshot?.gateway.pid ?? "none"}</strong>
                    </div>
                    <div>
                      <span>Log path</span>
                      <strong>
                        {compactValue(snapshot?.gateway.logPath, 18)}
                      </strong>
                    </div>
                    <div>
                      <span>Last pulse</span>
                      <strong>
                        {snapshot?.gateway.lastLogLine
                          ? parseLogLine(snapshot.gateway.lastLogLine).message
                          : "No live log line yet."}
                      </strong>
                    </div>
                  </div>
                </div>

                <div className="gateway-stage__control">
                  <div className="power-rack power-rack--gateway">
                    <button
                      className={
                        snapshot?.gateway.running
                          ? "power-core power-core--online"
                          : "power-core power-core--offline"
                      }
                      onClick={() => void handlePowerToggle()}
                      disabled={busyAction !== null || snapshot == null}
                      aria-label={
                        snapshot?.gateway.running
                          ? "Stop gateway"
                          : "Start gateway"
                      }
                    >
                      <span className="power-core__cluster" aria-hidden="true">
                        <img className="power-core__art" src={powerOrb} alt="" />
                      </span>
                    </button>

                    <div className="power-readout">
                      <span className="power-readout__label">{powerLabel}</span>
                      <span className="power-readout__hint">
                        {snapshot?.gateway.running
                          ? "Tap the orb to stop gateway"
                          : "Tap the orb to start gateway"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="gateway-stage__strip">
                {gatewayStageMetrics.map((tile) => (
                  <div key={tile.label} className="gateway-stage__datum">
                    <span>{tile.label}</span>
                    <strong>{tile.value}</strong>
                  </div>
                ))}
              </div>

              <div className="gateway-stage__routes">
                <div className="gateway-stage__routes-head">
                  <p className="eyebrow">Mission Lanes</p>
                  <span className="datum">
                    {snapshot?.factory.issues.length ?? 0} warnings
                  </span>
                </div>
                <div className="gateway-stage__routes-grid">
                  {missionCards.map((card) => (
                    <div key={card.label} className="gateway-stage__route">
                      <span>{card.label}</span>
                      <strong>{formatModel(card.value, "unset")}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </article>

            <section className="panel status-board">
              <div className="status-board__lane">
                <div className="label-row">
                  <p className="eyebrow">Auth Signal</p>
                  <HelpTip
                    label="Auth Signal"
                    text="Shows which OpenAI account is active and how long the current auth is expected to stay valid."
                    placement="bottom"
                  />
                </div>
                <h3>
                  <SensitiveValue
                    value={snapshot?.gateway.auth.activeAccount}
                    keep={16}
                  />
                </h3>
                <dl className="mini-facts">
                  <div>
                    <dt>Accounts</dt>
                    <dd>{snapshot?.gateway.auth.accountCount ?? 0}</dd>
                  </div>
                  <div>
                    <dt>Expiry horizon</dt>
                    <dd>
                      {formatMinutes(
                        snapshot?.gateway.auth.expiresInMinutes ?? null,
                      )}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="status-board__lane">
                  <p className="eyebrow">Droid Matrix</p>
                  <h3>{snapshot?.droids.length ?? 0} loaded</h3>
                  <dl className="mini-facts">
                    <div>
                      <dt>Explicit custom pins</dt>
                      <dd>{customDroidCount}</dd>
                    </div>
                    <div>
                      <dt>Needs attention</dt>
                      <dd>{issueCount}</dd>
                    </div>
                  </dl>
              </div>

              <div className="status-board__lane">
                <p className="eyebrow">Routing Health</p>
                <h3>{snapshot?.factory.issues.length ?? 0} warnings</h3>
                <dl className="mini-facts">
                  <div>
                    <dt>Inherited droids</dt>
                    <dd>{inheritedDroidCount}</dd>
                  </div>
                  <div>
                    <dt>Custom inventory</dt>
                    <dd>{snapshot?.factory.settingsCustomModelCount ?? 0}</dd>
                  </div>
                </dl>
              </div>
            </section>

            <article className="panel doctor-panel">
              <div className="panel-heading panel-heading--tight">
                <div>
                  <p className="eyebrow">Doctor Trace</p>
                  <h2>Runtime sweep</h2>
                </div>
                <div className="panel-action">
                  <button
                    className="button button--utility"
                    onClick={() => void handleDoctor()}
                    disabled={busyAction !== null}
                  >
                    {busyAction === "doctor" ? "Running..." : "Run Doctor"}
                  </button>
                  <HelpTip
                    label="Run Doctor"
                    text="Runs the built-in health sweep so you can see config drift, auth issues, and runtime problems."
                    placement="bottom"
                    align="end"
                  />
                </div>
              </div>
              <pre className="log-block log-block--compact">
                {doctorOutput || "No doctor run yet."}
              </pre>
            </article>
          </div>
        </section>

        <section
          className={activeTab === "droids" ? "view view--active" : "view"}
        >
          <div className="panel droids-command">
            <div className="filter-panel filter-panel--embedded">
              <div className="panel-heading">
                <div>
                  <div className="label-row">
                    <p className="eyebrow">Filter Deck</p>
                    <HelpTip
                      label="Filter Deck"
                      text="Search, preset, sort, and narrow the list without changing any droid pins."
                    />
                  </div>
                  <h3>Find the droid you need fast.</h3>
                </div>
                <button
                  className="button button--utility"
                  onClick={() => {
                    setSearchQuery("");
                    setDroidFilter("all");
                    setActivePreset("none");
                  }}
                  disabled={busyAction !== null}
                >
                  Reset
                </button>
              </div>
              <label className="search-field">
                <span>Search by name, path, or model</span>
                <input
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  placeholder="security, worker, xhigh..."
                />
              </label>
              <div className="preset-row">
                {DROID_PRESETS.map((preset) => (
                  <button
                    key={preset.key}
                    className={
                      activePreset === preset.key
                        ? "preset-pill preset-pill--active"
                        : "preset-pill"
                    }
                    onClick={() => applyPreset(preset.key)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <div className="filter-pill-row">
                {DROID_FILTERS.map((filter) => (
                  <button
                    key={filter.key}
                    className={
                      droidFilter === filter.key
                        ? "filter-pill filter-pill--active"
                        : "filter-pill"
                    }
                    onClick={() => setDroidFilter(filter.key)}
                  >
                    <span>{filter.label}</span>
                    <strong>{filterCounts[filter.key]}</strong>
                  </button>
                ))}
              </div>
              <div className="filter-panel__tools">
                <label className="filter-select">
                  <span>Sort</span>
                  <ThemedSelect
                    className="themed-select--field"
                    value={sortMode}
                    options={sortOptions}
                    onChange={(nextValue) =>
                      setSortMode(nextValue as DroidSortMode)
                    }
                  />
                </label>
                <button
                  className={
                    compactRows
                      ? "toggle-chip toggle-chip--active"
                      : "toggle-chip"
                  }
                  onClick={() => setCompactRows((current) => !current)}
                >
                  {compactRows ? "Compact rows on" : "Compact rows off"}
                </button>
              </div>
              <div className="filter-panel__summary">
                <span>
                  Showing {filteredDroids.length} of {droids.length} droids
                </span>
                <button
                  className="filter-link"
                  onClick={() =>
                    setCollapsedScopes({
                      workspace: false,
                      machine: false,
                    })
                  }
                >
                  Expand all
                </button>
              </div>
            </div>

            <div className="bulk-panel bulk-panel--embedded">
              <div className="panel-heading panel-heading--tight">
                <div className="bulk-panel__copy">
                  <div className="label-row">
                    <p className="eyebrow">Bulk Assignment</p>
                    <HelpTip
                      label="Bulk Assignment"
                      text="Apply the selected model to every visible droid or to a whole scope in one pass."
                    />
                  </div>
                  <h3>Retarget multiple droids in one pass.</h3>
                  <p>
                    Choose one model lane, then fan it out across the droids you
                    want to change. Factory defaults re-align automatically
                    after pin changes.
                  </p>
                </div>
              </div>
              <div className="bulk-panel__controls">
                <ThemedSelect
                  className="themed-select--field"
                  value={bulkModel}
                  options={bulkModelChoices}
                  onChange={setBulkModel}
                />
                <div className="bulk-panel__actions">
                  <button
                    className="button button--accent"
                    onClick={() =>
                      void handleBulkAssign(
                        "visible",
                        (droid) =>
                          filteredDroids.some(
                            (item) => item.path === droid.path,
                          ),
                        "Visible droids",
                      )
                    }
                    disabled={
                      busyAction !== null ||
                      !bulkModel ||
                      filteredDroids.length === 0
                    }
                  >
                    {busyAction === "bulk:visible"
                      ? "Applying..."
                      : "Apply visible"}
                  </button>
                  <button
                    className="button button--utility"
                    onClick={() =>
                      void handleBulkAssign("all", () => true, "All droids")
                    }
                    disabled={busyAction !== null || !bulkModel}
                  >
                    {busyAction === "bulk:all" ? "Applying..." : "Apply all"}
                  </button>
                  <button
                    className="button button--utility"
                    onClick={() =>
                      void handleBulkAssign(
                        "workspace",
                        (droid) => droid.scope === "workspace",
                        "Workspace droids",
                      )
                    }
                    disabled={busyAction !== null || !bulkModel}
                  >
                    {busyAction === "bulk:workspace"
                      ? "Applying..."
                      : "Workspace"}
                  </button>
                  <button
                    className="button button--utility"
                    onClick={() =>
                      void handleBulkAssign(
                        "machine",
                        (droid) => droid.scope === "machine",
                        "Machine droids",
                      )
                    }
                    disabled={busyAction !== null || !bulkModel}
                  >
                    {busyAction === "bulk:machine" ? "Applying..." : "Machine"}
                  </button>
                </div>
              </div>
              {bulkResult ? (
                <p className="bulk-panel__result">{bulkResult}</p>
              ) : null}
            </div>
          </div>

          <div
            className={
              compactRows
                ? "droid-columns droid-columns--compact"
                : "droid-columns"
            }
          >
            {(["workspace", "machine"] as const).map((scope) => (
              <article key={scope} className="panel droid-panel">
                <button
                  className="droid-panel__toggle"
                  onClick={() =>
                    setCollapsedScopes((current) => ({
                      ...current,
                      [scope]: !current[scope],
                    }))
                  }
                >
                  <div className="droid-panel__toggle-copy">
                    <p className="eyebrow">{scope}</p>
                    <h3>
                      {scope === "workspace" ? "Repo droids" : "Machine droids"}
                    </h3>
                  </div>
                  <div className="droid-panel__toggle-meta">
                    <span>
                      {groupedDroids[scope].length}/
                      {groupedAllDroids[scope].length} visible
                    </span>
                    <strong>{collapsedScopes[scope] ? "+" : "-"}</strong>
                  </div>
                </button>
                {!collapsedScopes[scope] ? (
                  <div className="droid-list">
                    {groupedDroids[scope].map((droid) => (
                      <div key={droid.path} className="droid-row">
                        <div className="droid-row__meta">
                          <div className="droid-row__topline">
                            <strong>{droid.name}</strong>
                            <span className={`chip chip--${droid.kind}`}>
                              {droid.kind}
                            </span>
                          </div>
                          <p className="droid-row__path">{droid.path}</p>
                          <p className="droid-row__model">
                            {formatModel(droid.model, "No model pin")}
                          </p>
                          {droid.issues.length > 0 ? (
                            <p className="droid-row__issue">
                              {droid.issues.join(" · ")}
                            </p>
                          ) : null}
                        </div>
                        <div className="droid-row__controls">
                          <ThemedSelect
                            className="themed-select--field"
                            value={
                              selectedModels[droid.path] ?? droid.model ?? ""
                            }
                            placeholder="Select model"
                            options={bulkModelChoices}
                            onChange={(nextValue) =>
                              setSelectedModels((current) => ({
                                ...current,
                                [droid.path]: nextValue,
                              }))
                            }
                          />
                          <button
                            className="button button--accent"
                            onClick={() => void handleModelSave(droid)}
                            disabled={
                              busyAction !== null ||
                              !selectedModels[droid.path] ||
                              selectedModels[droid.path] === droid.model
                            }
                          >
                            {busyAction === `save:${droid.path}`
                              ? "Saving..."
                              : "Apply Pin"}
                          </button>
                        </div>
                      </div>
                    ))}
                    {groupedDroids[scope].length === 0 ? (
                      <div className="empty-state">
                        No {scope} droids match the current filters.
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        </section>

        <section
          className={activeTab === "factory" ? "view view--active" : "view"}
        >
          <div className="factory-grid">
            <article className="panel factory-card factory-card--wide">
              <p className="eyebrow">Factory Blueprint</p>
              <h2>Config and settings anchors</h2>
              <dl className="fact-list">
                <div>
                  <dt>Factory home</dt>
                  <dd>{snapshot?.factory.homePath ?? "Unavailable"}</dd>
                </div>
                <div>
                  <dt>Legacy config</dt>
                  <dd>{snapshot?.factory.configPath ?? "Unavailable"}</dd>
                </div>
                <div>
                  <dt>Settings file</dt>
                  <dd>{snapshot?.factory.settingsPath ?? "Unavailable"}</dd>
                </div>
                <div>
                  <dt>Machine droids</dt>
                  <dd>{snapshot?.factory.machineDroidsPath ?? "Unavailable"}</dd>
                </div>
                <div>
                  <dt>Legacy custom models</dt>
                  <dd>{snapshot?.factory.legacyCustomModelCount ?? 0}</dd>
                </div>
                <div>
                  <dt>Settings custom models</dt>
                  <dd>{snapshot?.factory.settingsCustomModelCount ?? 0}</dd>
                </div>
              </dl>
            </article>

            <article className="panel factory-card">
              <p className="eyebrow">Session Spine</p>
              <h2>{snapshot?.factory.sessionDefaultModel ?? "Unset"}</h2>
              <p className="factory-card__copy">
                The current Factory default session model.
              </p>
            </article>

            <article className="panel factory-card">
              <p className="eyebrow">Mission Lanes</p>
              <h2>{missionCards.length}</h2>
              <p className="factory-card__copy">
                Orchestrator, worker, and validator routing are visible below.
              </p>
            </article>

            <article className="panel issues-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Drift Warnings</p>
                  <h2>What still needs attention</h2>
                </div>
                <span className="datum">
                  {snapshot?.factory.issues.length ?? 0}
                </span>
              </div>
              <ul>
                {(snapshot?.factory.issues ?? []).map((issue) => (
                  <li key={issue}>{issue}</li>
                ))}
                {(snapshot?.factory.issues.length ?? 0) === 0 ? (
                  <li>
                    Factory settings are aligned with your custom model
                    defaults.
                  </li>
                ) : null}
              </ul>
            </article>

            <article className="panel mission-panel mission-panel--factory">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Mission Routing</p>
                  <h2>Current system ids</h2>
                </div>
              </div>
              <div className="mission-panel__grid">
                {missionCards.map((card) => (
                  <div
                    key={card.label}
                    className="mission-card mission-card--factory"
                  >
                    <span>{card.label}</span>
                    <strong>{card.value}</strong>
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>

        <section className={activeTab === "acp" ? "view view--active" : "view"}>
          <div className="factory-grid">
            <article className="panel factory-card acp-stage">
              <div className="panel-heading panel-heading--tight">
                <div>
                  <p className="eyebrow">ACP Control Plane</p>
                  <h2>Adapter and runtime status</h2>
                </div>
                <span className={`health-pill health-pill--${acpHealthTone}`}>
                  {acpHealthLabel}
                </span>
              </div>

              {acpError ? (
                <div className="acp-inline-error">{acpError}</div>
              ) : null}

              {acpSnapshot ? (
                <>
                  <div className="acp-stage__strip">
                    {acpMetrics.map((metric) => (
                      <div key={metric.label} className="gateway-stage__datum">
                        <span>{metric.label}</span>
                        <strong>{metric.value}</strong>
                      </div>
                    ))}
                  </div>

                  <div className="acp-stage__fact-grid">
                    {acpFacts.map((fact) => (
                      <div key={fact.label} className="gateway-stage__datum">
                        <span>{fact.label}</span>
                        <strong>{fact.value}</strong>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  Loading ACP runtime status and recorded session summary.
                </div>
              )}
            </article>

            <article className="panel factory-card acp-adapters">
              <div className="panel-heading panel-heading--tight">
                <div>
                  <p className="eyebrow">Adapter Deck</p>
                  <h2>Ready runtimes and launch guidance</h2>
                </div>
              </div>

              <div className="acp-adapters__grid">
                {acpAgents.length > 0 ? (
                  acpAgents.map((agent) => (
                    <article key={agent.kind} className="acp-adapter-card">
                      <div className="acp-adapter-card__head">
                        <div>
                          <p className="eyebrow">{agent.kind}</p>
                          <h3>{agent.runtimeName}</h3>
                        </div>
                        <span
                          className={`health-pill health-pill--${getAgentTone(agent)}`}
                        >
                          {agent.ready ? "Ready" : "Unavailable"}
                        </span>
                      </div>
                      <p className="acp-adapter-card__status">
                        {formatSessionEvent(agent.status)}
                      </p>
                      {agent.issue ? (
                        <p className="acp-adapter-card__issue">{agent.issue}</p>
                      ) : null}
                      {agent.guidance.length > 0 ? (
                        <ul className="acp-guidance-list">
                          {agent.guidance.map((line) => (
                            <li key={line}>{line}</li>
                          ))}
                        </ul>
                      ) : null}
                    </article>
                  ))
                ) : (
                  <div className="empty-state">
                    No ACP adapters have reported readiness yet.
                  </div>
                )}
              </div>
            </article>

            <article className="panel factory-card acp-issues">
              <div className="panel-heading panel-heading--tight">
                <div>
                  <p className="eyebrow">Issue Feed</p>
                  <h2>Recent ACP issues</h2>
                </div>
                <span className="datum">{acpIssues.length}</span>
              </div>

              <div className="acp-issues__list">
                {acpIssues.length > 0 ? (
                  acpIssues.map((issue) => (
                    <article
                      key={`${issue.scope}:${issue.label}:${issue.timestampMs ?? "none"}:${issue.message}`}
                      className="acp-issue-card"
                    >
                      <div className="acp-issue-card__head">
                        <div className="acp-issue-card__title">
                          <span className="chip chip--flagged">
                            {formatSessionEvent(issue.scope)}
                          </span>
                          <strong>{issue.label}</strong>
                          {issue.agentKind ? (
                            <span className="chip chip--inherit">{issue.agentKind}</span>
                          ) : null}
                        </div>
                        <span className="acp-issue-card__time">
                          {formatSessionTimestamp(issue.timestampMs)}
                        </span>
                      </div>
                      <p className="acp-issue-card__message">{issue.message}</p>
                      <div className="acp-issue-card__facts">
                        {issue.sessionId ? (
                          <span>session {issue.sessionId}</span>
                        ) : null}
                        {issue.cwd ? <span>{compactValue(issue.cwd, 18)}</span> : null}
                      </div>
                    </article>
                  ))
                ) : (
                  <div className="empty-state">
                    No recent ACP issues are recorded right now.
                  </div>
                )}
              </div>
            </article>

            <article className="panel factory-card factory-card--wide acp-sessions">
              <div className="panel-heading panel-heading--tight">
                <div>
                  <p className="eyebrow">Session Ledger</p>
                  <h2>Recorded ACP sessions</h2>
                </div>
                <span className="datum">
                  {acpSnapshot?.metrics.sessionsCreated ?? 0} created
                </span>
              </div>

              <div className="acp-session-list">
                {acpSessions.length > 0 ? (
                  acpSessions.map((session) => (
                    <button
                      key={session.sessionId}
                      type="button"
                      className={
                        session.sessionId === selectedAcpSessionId
                          ? "acp-session-row acp-session-row--active"
                          : "acp-session-row"
                      }
                      onClick={() => {
                        setSelectedAcpSessionId(session.sessionId);
                        setAcpDetailNotice(null);
                        void refreshAcpSessionDetail(session.sessionId, acpDetailLimit);
                      }}
                    >
                      <div className="acp-session-row__main">
                        <div className="acp-session-row__topline">
                          <strong>{session.sessionId}</strong>
                          {session.agentKind ? (
                            <span className="chip chip--inherit">
                              {session.agentKind}
                            </span>
                          ) : null}
                          <span className="chip chip--custom">
                            {formatSessionState(session.state)}
                          </span>
                          <span className="chip chip--custom">
                            {session.promptCount} prompts
                          </span>
                          <span className="chip chip--inherit">
                            {formatSessionEvent(session.lastEvent)}
                          </span>
                        </div>
                        <p className="droid-row__path">
                          {session.cwd ?? "No working directory recorded"}
                        </p>
                      </div>

                      <dl className="acp-session-row__facts">
                        <div>
                          <dt>Started</dt>
                          <dd>{formatSessionTimestamp(session.startedTimestampMs)}</dd>
                        </div>
                        <div>
                          <dt>Last seen</dt>
                          <dd>{formatSessionTimestamp(session.lastTimestampMs)}</dd>
                        </div>
                        <div>
                          <dt>Log</dt>
                          <dd>{compactValue(session.logPath, 18)}</dd>
                        </div>
                      </dl>
                    </button>
                  ))
                ) : (
                  <div className="empty-state">
                    No ACP sessions have been recorded yet.
                  </div>
                )}
              </div>
            </article>

            <article className="panel factory-card factory-card--wide acp-detail">
              <div className="panel-heading panel-heading--tight">
                <div>
                  <p className="eyebrow">Session Detail</p>
                  <h2>
                    {selectedAcpSession?.sessionId ??
                      acpSessionDetail?.summary.sessionId ??
                      "Choose a session"}
                  </h2>
                </div>
                {acpSessionDetail ? (
                  <span className="datum">
                    {acpSessionDetail.metrics.promptsCompleted} completed
                  </span>
                ) : null}
              </div>

              {acpDetailError ? (
                <div className="acp-inline-error">{acpDetailError}</div>
              ) : null}
              <div className="acp-detail__toolbar">
                <div className="acp-detail__toolbar-group">
                  {[12, 40, 120].map((limit) => (
                    <button
                      key={limit}
                      type="button"
                      className={
                        limit === acpDetailLimit
                          ? "toggle-chip toggle-chip--active"
                          : "toggle-chip"
                      }
                      onClick={() => {
                        setAcpDetailLimit(limit);
                        if (selectedAcpSessionId) {
                          void refreshAcpSessionDetail(selectedAcpSessionId, limit);
                        }
                      }}
                    >
                      Tail {limit}
                    </button>
                  ))}
                </div>
                <div className="acp-detail__toolbar-group">
                  <button
                    type="button"
                    className="button button--utility"
                    disabled={!acpSessionDetail}
                    onClick={() =>
                      acpSessionDetail
                        ? void copyAcpPath(
                            "Journal path",
                            acpSessionDetail.summary.journalPath,
                          )
                        : undefined
                    }
                  >
                    Copy journal path
                  </button>
                  <button
                    type="button"
                    className="button button--utility"
                    disabled={!acpSessionDetail}
                    onClick={() =>
                      acpSessionDetail
                        ? void copyAcpPath("Log path", acpSessionDetail.summary.logPath)
                        : undefined
                    }
                  >
                    Copy log path
                  </button>
                </div>
              </div>
              {acpDetailNotice ? (
                <p className="acp-detail__notice">{acpDetailNotice}</p>
              ) : null}

              {acpSessionDetail ? (
                <>
                  <div className="acp-detail__metrics">
                    <div className="gateway-stage__datum">
                      <span>Completed</span>
                      <strong>{acpSessionDetail.metrics.promptsCompleted}</strong>
                    </div>
                    <div className="gateway-stage__datum">
                      <span>Cancelled</span>
                      <strong>{acpSessionDetail.metrics.promptsCancelled}</strong>
                    </div>
                    <div className="gateway-stage__datum">
                      <span>Runtime failures</span>
                      <strong>{acpSessionDetail.metrics.runtimeFailures}</strong>
                    </div>
                    <div className="gateway-stage__datum">
                      <span>Working dir</span>
                      <strong>
                        {acpSessionDetail.summary.cwd ?? "No working directory recorded"}
                      </strong>
                    </div>
                  </div>

                  <div className="acp-detail__grid">
                    <section className="acp-detail__panel">
                      <div className="label-row">
                        <p className="eyebrow">Status Timeline</p>
                      </div>
                      <div className="acp-event-list">
                        {acpSessionDetail.recentEvents.length > 0 ? (
                          acpSessionDetail.recentEvents.map((event, index) => (
                            <article
                              key={`${event.timestampMs ?? "none"}:${event.event ?? "unknown"}:${index}`}
                              className="acp-event-row"
                            >
                              <div className="acp-event-row__meta">
                                <div className="acp-event-row__track">
                                  <span className="acp-event-row__dot" />
                                  <span className="chip chip--inherit">
                                    {formatSessionEvent(event.event)}
                                  </span>
                                </div>
                                <span className="acp-event-row__time">
                                  {formatSessionTimestamp(event.timestampMs)}
                                </span>
                              </div>
                              <pre className="acp-event-row__data">
                                {event.dataPreview}
                              </pre>
                            </article>
                          ))
                        ) : (
                          <div className="empty-state">
                            No recent ACP events were recorded for this session.
                          </div>
                        )}
                      </div>
                    </section>

                    <section className="acp-detail__panel">
                      <div className="label-row">
                        <p className="eyebrow">Recent Log Lines</p>
                      </div>
                      <div className="log-block log-block--compact acp-detail__logs">
                        {acpSessionDetail.recentLogs.length > 0 ? (
                          acpSessionDetail.recentLogs.map((line, index) => (
                            <div key={`${line}:${index}`} className="acp-detail__log-line">
                              {line}
                            </div>
                          ))
                        ) : (
                          <div className="empty-state">
                            No recent ACP log lines were recorded for this session.
                          </div>
                        )}
                      </div>
                    </section>
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  Select a recorded ACP session to inspect recent events and logs.
                </div>
              )}
            </article>
          </div>
        </section>

        <section
          className={activeTab === "logs" ? "view view--active" : "view"}
        >
          <div className="panel logs-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Log Tail</p>
                <h2>Live gateway output</h2>
              </div>
              <button
                className="button button--utility"
                onClick={() =>
                  void runAction(
                    "logs",
                    () => call<string[]>("tail_logs", { limit: 220 }),
                    (rows) => setLogs(rows as string[]),
                  )
                }
                disabled={busyAction !== null}
              >
                Refresh Tail
              </button>
            </div>
            <div className="logs-panel__strip">
              <span className="status-led status-led--live" />
              <span>
                {snapshot?.gateway.logPath ?? "No log path available"}
              </span>
              <span className="logs-panel__mode">Newest first</span>
            </div>
            <div className="log-feed log-block log-block--tall">
              {logEntries.length > 0 ? (
                logEntries.map((entry, index) => (
                  <div className="log-row" key={`${entry.raw}:${index}`}>
                    <span className="log-row__time">
                      {formatLogTimestamp(entry.timestamp)}
                    </span>
                    <span className="log-row__message">{entry.message}</span>
                  </div>
                ))
              ) : (
                <div className="log-row log-row--empty">
                  <span className="log-row__message">
                    No log output captured yet.
                  </span>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
