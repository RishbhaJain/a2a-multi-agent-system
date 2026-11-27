import asyncio
import unittest

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

from a2a_simple.__main__ import RouterAgentExecutor


class ImmediateAgent:
    async def invoke(self, *_args, **_kwargs) -> str:
        return "completed result"


class BlockingAgent:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def invoke(self, *_args, **_kwargs) -> str:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def request(task_id: str, context_id: str) -> MessageSendParams:
    return MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="Calculate 2 + 2"))],
            messageId="message-1",
            taskId=task_id,
            contextId=context_id,
        )
    )


class A2ATaskLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_executor_emits_working_and_completed_task_states(self) -> None:
        executor = RouterAgentExecutor(agent=ImmediateAgent())
        queue = EventQueue()

        await executor.execute(
            RequestContext(task_id="task-1", context_id="context-1"),
            queue,
        )

        working = await queue.dequeue_event(no_wait=True)
        completed = await queue.dequeue_event(no_wait=True)
        self.assertIsInstance(working, TaskStatusUpdateEvent)
        self.assertEqual(working.status.state, TaskState.working)
        self.assertFalse(working.final)
        self.assertIsInstance(completed, TaskStatusUpdateEvent)
        self.assertEqual(completed.status.state, TaskState.completed)
        self.assertTrue(completed.final)
        self.assertEqual(completed.status.message.taskId, "task-1")
        self.assertEqual(completed.status.message.contextId, "context-1")
        self.assertEqual(
            completed.status.message.parts[0].root.text,
            "completed result",
        )

    async def test_a2a_cancel_stops_running_agent_and_persists_state(self) -> None:
        agent = BlockingAgent()
        store = InMemoryTaskStore()
        handler = DefaultRequestHandler(
            agent_executor=RouterAgentExecutor(agent=agent),
            task_store=store,
        )
        send_task = asyncio.create_task(
            handler.on_message_send(request("task-2", "context-2"))
        )
        await asyncio.wait_for(agent.entered.wait(), timeout=1)

        for _ in range(20):
            active = await store.get("task-2")
            if active is not None and active.status.state == TaskState.working:
                break
            await asyncio.sleep(0)
        else:
            self.fail("working task was not persisted before cancellation")

        canceled = await asyncio.wait_for(
            handler.on_cancel_task(TaskIdParams(id="task-2")),
            timeout=1,
        )
        original_result = await asyncio.wait_for(send_task, timeout=1)

        self.assertIsInstance(canceled, Task)
        self.assertEqual(canceled.status.state, TaskState.canceled)
        self.assertIsInstance(original_result, Task)
        self.assertEqual(original_result.status.state, TaskState.canceled)
        self.assertTrue(agent.cancelled.is_set())
        persisted = await store.get("task-2")
        self.assertEqual(persisted.status.state, TaskState.canceled)


if __name__ == "__main__":
    unittest.main()
