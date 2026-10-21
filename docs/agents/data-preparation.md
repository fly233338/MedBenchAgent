# Data Preparation Agent

**Input:** frozen inventory for one dataset's one-or-more sources, accepted upstream artifacts, user decisions and host validation feedback.

**Output:** Understand mapping/conversion policy; Text Cases/accounting; conditional Image converted candidates; host-published immutable Prepared revision.

**Private state:** preparation checkpoint, stage/session epoch, round counter, failure counters, accepted handoff, questions and candidate snapshots.

**Model-visible context:** stage instructions, source inventory references, accepted semantic inputs, current feedback and explicit user decisions. Raw source files remain read-only. The worker uses OpenHands terminal/file tools; the host owns acceptance and publication.

**Conversation:** each stage/semantic epoch has one stable workspace and conversation ID. Ordinary implementation repairs and answers append a deduplicated host message to that session. Every completed worker round has a separate candidate snapshot. A mapping revision starts a fresh epoch and invalidates dependent Text/Image work.

**Feedback and budgets:** the first rejected candidate establishes a baseline of content digest and Host-passed checks. A changed candidate with improved checks or resolved diagnostics resets consecutive no-progress failures; cosmetic rewrites, repeated errors and regressions consume the allowance. User questions do not consume it. Stage iterations and active execution time are cumulative across repairs and resume; waiting for answers is excluded. Explicit blocked outcomes and environment failures stop for intervention. Mapping revisions have their own finite budget. SDK condensation manages growing conversation context; accepted semantics remain in explicit files and subsequent host messages.

**Recovery:** reuse a completed round receipt, or restore the unfinished SDK conversation. The SDK event history is checked before sending a host-turn marker again. The source inventory, handoff and previous candidate references remain digest-bound.

**Acceptance:** host checks required files, content, population accounting and images; worker completion alone is insufficient. Tests cover same-session retry, clarification, immutable snapshots, mapping revision, conversion skipping and control precedence. Live Docker/model throughput remains separate validation.

See the [stage validation contract](../PREPARATION_VALIDATION.md).
