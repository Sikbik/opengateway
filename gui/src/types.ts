export type HealthState = "online" | "degraded" | "offline";

export interface AppSnapshot {
  generatedAt: number;
  workspacePath: string | null;
  environment: EnvironmentSnapshot;
  gateway: GatewaySnapshot;
  factory: FactorySnapshot;
  models: ModelOption[];
  droids: DroidRecord[];
}

export interface EnvironmentSnapshot {
  os: string;
  isWsl: boolean;
  desktopShellSupported: boolean;
  preferredRuntime: string;
}

export interface GatewaySnapshot {
  running: boolean;
  pid: number | null;
  health: HealthState;
  apiBaseUrl: string;
  logPath: string;
  preferredModel: string | null;
  auth: AuthSnapshot;
  lastLogLine: string | null;
}

export interface AuthSnapshot {
  accountCount: number;
  activeAccount: string | null;
  expiresAtMs: number | null;
  expiresInMinutes: number | null;
}

export interface FactorySnapshot {
  homePath: string;
  configPath: string;
  settingsPath: string;
  machineDroidsPath: string;
  legacyCustomModelCount: number;
  settingsCustomModelCount: number;
  sessionDefaultModel: string | null;
  missionModels: MissionModels;
  issues: string[];
}

export interface MissionModels {
  orchestrator: string | null;
  worker: string | null;
  validationWorker: string | null;
}

export interface ModelOption {
  displayName: string;
  model: string;
  id: string | null;
  source: string;
}

export interface DroidRecord {
  name: string;
  path: string;
  scope: "machine" | "workspace";
  model: string | null;
  kind: "custom" | "inherit" | "builtin" | "missing";
  issues: string[];
}

export interface CommandResult {
  success: boolean;
  output: string;
}

export interface AcpGuiSnapshot {
  experimental: boolean;
  command: string;
  transport: string;
  processModel: string;
  metrics: AcpGuiMetrics;
  agents: AcpGuiAgent[];
  issues: AcpGuiIssue[];
  sessions: AcpGuiSession[];
}

export interface AcpGuiMetrics {
  sessionsCreated: number;
  promptsCompleted: number;
  promptsCancelled: number;
  runtimeFailures: number;
}

export interface AcpGuiAgent {
  kind: string;
  runtimeName: string;
  status: string;
  ready: boolean;
  issue: string | null;
  guidance: string[];
}

export interface AcpGuiSession {
  sessionId: string;
  agentKind: string | null;
  state: string;
  promptCount: number;
  cwd: string | null;
  startedTimestampMs: number | null;
  lastEvent: string | null;
  lastTimestampMs: number | null;
  journalPath: string;
  logPath: string;
}

export interface AcpGuiSessionDetail {
  summary: AcpGuiSession;
  metrics: AcpGuiMetrics;
  recentEvents: AcpGuiSessionEvent[];
  recentLogs: string[];
}

export interface AcpGuiSessionEvent {
  timestampMs: number | null;
  event: string | null;
  dataPreview: string;
}

export interface AcpGuiIssue {
  scope: string;
  severity: string;
  label: string;
  message: string;
  hint: string | null;
  sessionId: string | null;
  agentKind: string | null;
  cwd: string | null;
  timestampMs: number | null;
}
