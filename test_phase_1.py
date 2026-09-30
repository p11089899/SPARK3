"""
PHASE 1 PASS CONDITION TEST
─────────────────────────────────────────────────────────────────────────────
Two dummy agents — ALPHA and BETA — are registered with the kernel.
ALPHA publishes a USER_INPUT event during start().
BETA subscribes to USER_INPUT and records every event it receives.

PASS:  BETA receives the event published by ALPHA.
FAIL:  BETA receives nothing, or ALPHA imports BETA directly (checked below).

Neither agent imports the other. All communication is through the EventBus.
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import inspect
import pytest

from core import (
    SparkKernel,
    BaseModule,
    EventBus,
    EventType,
    SparkEvent,
    ModuleState,
)


# ── dummy agents ──────────────────────────────────────────────────────────────

class AlphaAgent(BaseModule):
    """Publishes one USER_INPUT event during start(). Knows nothing about BETA."""

    name         = "ALPHA"
    version      = "0.1.0"
    dependencies: list[str] = []

    async def start(self) -> None:
        await self.bus.publish(SparkEvent(
            type    = EventType.USER_INPUT,
            source  = "ALPHA",
            payload = {"text": "hello from alpha"},
        ))

    async def stop(self) -> None:
        pass


class BetaAgent(BaseModule):
    """Subscribes to USER_INPUT. Records every event it receives. Knows nothing about ALPHA."""

    name         = "BETA"
    version      = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, bus: EventBus, config: dict) -> None:
        super().__init__(bus, config)
        self.received: list[SparkEvent] = []

    async def start(self) -> None:
        self.bus.subscribe(EventType.USER_INPUT, self._on_input)

    async def stop(self) -> None:
        self.bus.unsubscribe(EventType.USER_INPUT, self._on_input)

    async def _on_input(self, event: SparkEvent) -> None:
        self.received.append(event)


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_event_flows_between_agents() -> None:
    """BETA receives the event published by ALPHA via the EventBus."""
    kernel = SparkKernel()
    kernel.register(AlphaAgent)
    kernel.register(BetaAgent)

    await kernel.boot()

    # Give the dispatch loop time to deliver the event
    await asyncio.sleep(0.1)

    beta: BetaAgent = kernel.registry["BETA"]   # type: ignore[assignment]

    assert len(beta.received) >= 1, (
        "BETA received no events. EventBus dispatch loop is not delivering."
    )

    event = beta.received[0]
    assert event.type   == EventType.USER_INPUT
    assert event.source == "ALPHA"
    assert event.payload["text"] == "hello from alpha"

    await kernel.shutdown()


@pytest.mark.asyncio
async def test_no_direct_imports_between_agents() -> None:
    """
    AlphaAgent's source code must not import BetaAgent, and vice versa.
    This enforces Law §02: agents never import each other.
    """
    alpha_src = inspect.getsource(AlphaAgent)
    beta_src  = inspect.getsource(BetaAgent)

    assert "BetaAgent"  not in alpha_src, "ALPHA imports BETA — Law §02 violation"
    assert "AlphaAgent" not in beta_src,  "BETA imports ALPHA — Law §02 violation"


@pytest.mark.asyncio
async def test_kernel_topo_sort_respects_dependencies() -> None:
    """Agent with no dependencies boots before agents that depend on it."""

    class RootAgent(BaseModule):
        name         = "ROOT"
        dependencies: list[str] = []
        boot_time:   float = 0.0

        async def start(self) -> None:
            import time
            self.__class__.boot_time = time.time()

        async def stop(self) -> None:
            pass

    class ChildAgent(BaseModule):
        name         = "CHILD"
        dependencies = ["ROOT"]
        boot_time:   float = 0.0

        async def start(self) -> None:
            import time
            self.__class__.boot_time = time.time()

        async def stop(self) -> None:
            pass

    kernel = SparkKernel()
    kernel.register(ChildAgent)   # registered first intentionally
    kernel.register(RootAgent)    # registered second

    await kernel.boot()

    assert RootAgent.boot_time  > 0
    assert ChildAgent.boot_time > 0
    assert RootAgent.boot_time <= ChildAgent.boot_time, (
        "CHILD booted before ROOT — topo sort is broken"
    )

    await kernel.shutdown()


@pytest.mark.asyncio
async def test_kernel_detects_unknown_dependency() -> None:
    """Kernel must raise ValueError if a dependency is not registered."""

    class OrphanAgent(BaseModule):
        name         = "ORPHAN"
        dependencies = ["NONEXISTENT_AGENT"]

        async def start(self) -> None: pass
        async def stop(self)  -> None: pass

    kernel = SparkKernel()
    kernel.register(OrphanAgent)

    with pytest.raises((ValueError, RuntimeError)):
        await kernel.boot()


@pytest.mark.asyncio
async def test_agent_state_transitions() -> None:
    """Agent moves through REGISTERED → LOADING → ACTIVE on boot."""

    states_seen: list[str] = []

    class WatchedAgent(BaseModule):
        name         = "WATCHED"
        dependencies: list[str] = []

        async def start(self) -> None:
            states_seen.append(self.state.value)  # should be LOADING here

        async def stop(self) -> None:
            pass

    kernel = SparkKernel()
    kernel.register(WatchedAgent)

    agent: WatchedAgent = kernel.registry["WATCHED"]   # type: ignore[assignment]
    assert agent.state == ModuleState.REGISTERED

    await kernel.boot()

    assert ModuleState.LOADING.value in states_seen, "Agent never entered LOADING state"
    assert agent.state == ModuleState.ACTIVE,         "Agent is not ACTIVE after boot"

    await kernel.shutdown()

    assert agent.state == ModuleState.STOPPED


@pytest.mark.asyncio
async def test_eventbus_subscriber_count() -> None:
    """subscribe() and unsubscribe() correctly adjust handler counts."""
    bus = EventBus()

    async def handler_a(e: SparkEvent) -> None: pass
    async def handler_b(e: SparkEvent) -> None: pass

    bus.subscribe(EventType.USER_INPUT, handler_a)
    bus.subscribe(EventType.USER_INPUT, handler_b)
    assert bus.subscriber_count(EventType.USER_INPUT) == 2

    bus.unsubscribe(EventType.USER_INPUT, handler_a)
    assert bus.subscriber_count(EventType.USER_INPUT) == 1

    bus.unsubscribe(EventType.USER_INPUT, handler_a)   # silent no-op
    assert bus.subscriber_count(EventType.USER_INPUT) == 1


@pytest.mark.asyncio
async def test_spark_ready_event_fires_after_boot() -> None:
    """Kernel publishes SPARK_READY after all agents are online."""
    received: list[SparkEvent] = []

    class ListenerAgent(BaseModule):
        name         = "LISTENER"
        dependencies: list[str] = []

        async def start(self) -> None:
            self.bus.subscribe(EventType.SPARK_READY, self._on_ready)

        async def stop(self) -> None:
            self.bus.unsubscribe(EventType.SPARK_READY, self._on_ready)

        async def _on_ready(self, event: SparkEvent) -> None:
            received.append(event)

    kernel = SparkKernel()
    kernel.register(ListenerAgent)
    await kernel.boot()
    await asyncio.sleep(0.1)

    assert len(received) == 1
    assert received[0].type   == EventType.SPARK_READY
    assert received[0].source == "kernel"

    await kernel.shutdown()