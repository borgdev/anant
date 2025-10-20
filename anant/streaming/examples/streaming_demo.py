"""
Streaming Framework Example
===========================

Demonstrates the comprehensive streaming framework with:
- Real-time event processing
- Persistent event storage
- Temporal graph analysis
- WebSocket connectivity
- Stream analytics
"""

import asyncio
import time
from datetime import datetime, timedelta
from anant.streaming import (
    create_real_time_streaming, StreamingConfig, StreamingMode,
    GraphEvent, EventType, TimeRange
)


async def streaming_example():
    """Demonstrate the streaming framework capabilities."""
    print("🚀 Starting Streaming Framework Example")
    
    # Create streaming framework with real-time configuration
    config = StreamingConfig(
        mode=StreamingMode.REAL_TIME,
        enable_persistence=True,
        event_store_backend="sqlite",
        event_store_path="example_events.db",
        enable_temporal_analysis=True,
        enable_websocket=True,
        websocket_port=8765,
        enable_metrics=True
    )
    
    framework = await create_real_time_streaming(
        backend="sqlite",
        enable_websocket=True,
        event_store_path="example_events.db"
    )
    
    # Start the framework
    async with framework.streaming_session():
        print("✅ Streaming framework started")
        print(f"📊 WebSocket server available at ws://localhost:8765")
        
        # Simulate graph events
        events = [
            GraphEvent(
                event_id="evt_1",
                event_type=EventType.EDGE_ADDED,
                timestamp=datetime.now(),
                graph_id="demo_graph",
                data={
                    "edge_id": "e1",
                    "nodes": ["n1", "n2", "n3"],
                    "weight": 1.0
                },
                source="demo"
            ),
            GraphEvent(
                event_id="evt_2", 
                event_type=EventType.EDGE_ADDED,
                timestamp=datetime.now() + timedelta(seconds=1),
                graph_id="demo_graph",
                data={
                    "edge_id": "e2",
                    "nodes": ["n2", "n3", "n4"],
                    "weight": 0.8
                },
                source="demo"
            ),
            GraphEvent(
                event_id="evt_3",
                event_type=EventType.NODE_ADDED,
                timestamp=datetime.now() + timedelta(seconds=2),
                graph_id="demo_graph",
                data={
                    "node_id": "n5",
                    "properties": {"type": "user"}
                },
                source="demo"
            )
        ]
        
        # Process events
        print("📥 Processing graph events...")
        for event in events:
            success = await framework.process_event(event)
            if success:
                print(f"   ✓ Processed {event.event_type.value}: {event.event_id}")
            else:
                print(f"   ✗ Failed to process: {event.event_id}")
            
            # Small delay between events
            await asyncio.sleep(0.5)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Get streaming statistics
        stats = await framework.get_stats()
        print(f"\n📈 Streaming Statistics:")
        print(f"   Events processed: {stats.events_processed}")
        print(f"   Events per second: {stats.events_per_second:.2f}")
        print(f"   Storage size: {stats.storage_size_mb:.2f} MB")
        print(f"   Active WebSocket connections: {stats.active_websockets}")
        print(f"   Uptime: {stats.uptime_seconds:.1f} seconds")
        
        # Query historical events
        from anant.streaming.core.event_store import EventQuery
        query = EventQuery(
            graph_id="demo_graph",
            event_types=[EventType.EDGE_ADDED],
            limit=10
        )
        
        historical_events = await framework.query_events(query)
        print(f"\n🔍 Historical Events Query:")
        print(f"   Found {len(historical_events)} edge addition events")
        
        for event in historical_events:
            print(f"   - {event.timestamp}: {event.data.get('edge_id')} -> {event.data.get('nodes')}")
        
        # Demonstrate temporal analysis if available
        if framework.temporal_graph:
            print(f"\n⏰ Temporal Analysis:")
            temporal_stats = framework.temporal_graph.get_storage_stats()
            print(f"   Snapshots: {temporal_stats.get('num_snapshots', 0)}")
            print(f"   Current version: {temporal_stats.get('current_version', 0)}")
        
        # Wait a bit to allow WebSocket connections
        print(f"\n🌐 WebSocket server is running. You can connect to:")
        print(f"   ws://localhost:8765")
        print(f"   Try connecting with a WebSocket client!")
        print(f"   Send: {{'type': 'subscribe', 'graph_id': 'demo_graph'}}")
        print(f"   Send: {{'type': 'stats'}}")
        print(f"   Send: {{'type': 'query', 'query': {{'graph_id': 'demo_graph', 'limit': 5}}}}")
        
        # Keep running for a bit to allow WebSocket testing
        print(f"\n⏳ Keeping server running for 30 seconds...")
        await asyncio.sleep(30)
        
        print(f"\n🏁 Streaming example completed!")


async def event_store_example():
    """Demonstrate event store capabilities."""
    print("\n📦 Event Store Example")
    
    from anant.streaming.core.event_store import create_event_store
    
    # Create event store
    store = await create_event_store("sqlite", db_path="event_store_example.db")
    
    # Create sample events
    events = []
    for i in range(10):
        event = GraphEvent(
            event_id=f"store_evt_{i}",
            event_type=EventType.EDGE_ADDED if i % 2 == 0 else EventType.NODE_ADDED,
            timestamp=datetime.now() + timedelta(seconds=i),
            graph_id="store_demo",
            data={"item_id": f"item_{i}", "value": i * 10},
            source="store_demo"
        )
        events.append(event)
    
    # Store events
    stored = await store.store_events(events)
    print(f"   Stored {stored} events")
    
    # Query events
    from anant.streaming.core.event_store import EventQuery
    query = EventQuery(graph_id="store_demo", limit=5)
    results = await store.query_events(query)
    print(f"   Queried {len(results)} events")
    
    # Replay events
    print(f"   Replaying events:")
    async for event in store.replay_events("store_demo"):
        print(f"      {event.timestamp}: {event.event_type.value} - {event.data}")
    
    # Get statistics
    stats = await store.get_stats()
    print(f"   Store stats: {stats}")
    
    await store.close()
    print(f"   Event store example completed")


async def temporal_graph_example():
    """Demonstrate temporal graph capabilities."""
    print("\n⏰ Temporal Graph Example")
    
    from anant.streaming.core.temporal_graph import TemporalGraph, TemporalScope, TimeRange
    from anant.classes.hypergraph import Hypergraph
    import polars as pl
    
    # Create temporal graph
    temporal_graph = TemporalGraph()
    
    # Create sample hypergraphs for different time points
    timestamps = [
        datetime.now() - timedelta(hours=2),
        datetime.now() - timedelta(hours=1),
        datetime.now()
    ]
    
    for i, timestamp in enumerate(timestamps):
        # Create hypergraph with evolving structure
        data = pl.DataFrame({
            "edges": [f"e{j}" for j in range(i + 1) for _ in range(2)],
            "nodes": [f"n{j}" for j in range(i + 1) for _ in range(2)],
            "weight": [1.0] * (2 * (i + 1))
        })
        
        hg = Hypergraph(data=data)
        
        # Add snapshot
        version = temporal_graph.add_snapshot(
            hg, 
            timestamp=timestamp,
            metadata={"description": f"Graph state at time {i}"}
        )
        print(f"   Added snapshot version {version} at {timestamp}")
    
    # Query temporal data
    time_range = TimeRange(
        start=datetime.now() - timedelta(hours=3),
        end=datetime.now()
    )
    
    snapshots = temporal_graph.get_snapshots_in_range(time_range)
    print(f"   Found {len(snapshots)} snapshots in time range")
    
    # Analyze evolution
    evolution = temporal_graph.compute_graph_evolution_metrics(time_range)
    print(f"   Evolution metrics: {evolution}")
    
    # Storage stats
    storage_stats = temporal_graph.get_storage_stats()
    print(f"   Storage stats: {storage_stats}")
    
    print(f"   Temporal graph example completed")


async def main():
    """Run all streaming examples."""
    print("🎯 Anant Streaming Framework Examples")
    print("=" * 50)
    
    try:
        # Run examples
        await streaming_example()
        await event_store_example()
        await temporal_graph_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Real-time event processing")
        print("✓ Persistent event storage (SQLite)")
        print("✓ WebSocket real-time connectivity")
        print("✓ Temporal graph analysis")
        print("✓ Event replay and querying")
        print("✓ Streaming analytics and metrics")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())