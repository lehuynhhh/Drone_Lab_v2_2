"""Tests for actual online navigation, not a precomputed route."""
import unittest
from time import monotonic
from types import SimpleNamespace
from drone_sim_3d_2 import (Drone,ObstacleManager,FrontSensor,FlightController,
    ReactiveNavigator,PerceptionSensor,SensorReading,Minimap,FlightUI,V,STEP,
    map_coordinates,BRAKE,RADIUS)


def world():
    d,m,s=Drone(),ObstacleManager(),FrontSensor()
    d.p.y=2; d.takeoff()
    return d,m,s,FlightController(d,m,s)


class ReactiveTests(unittest.TestCase):
    def fly(self,d,c,limit=9000):
        for _ in range(limit):
            c.tick(STEP)
            self.assertFalse(c.collision,f'Collision at {d.p}')
            if c.mission=='ARRIVED': return
        self.fail(f'Not arrived: {c.mission}, {d.p}, {c.navigator.events}')

    def test_empty_world_is_straight(self):
        d,m,s,c=world(); c.go(V(0,2,12))
        self.fly(d,c)
        self.assertEqual(c.navigator.events,[])
        self.assertAlmostEqual(d.p.x,0,places=8)

    def test_go_does_not_read_map_or_compute_route(self):
        class Guard:
            @property
            def items(self): raise AssertionError('Navigator accessed world obstacle list')
            def sweep(self,*args): return 1
        d=Drone(); d.p.y=2
        sensor=SimpleNamespace(reading=SensorReading(),scan=lambda *a:SensorReading())
        c=FlightController(d,Guard(),sensor)
        c.go(V(0,2,12))
        self.assertFalse(hasattr(c,'route'))
        self.assertFalse(c.navigator.memory)
        for _ in range(120): c.tick(STEP)
        self.assertEqual(c.navigator.state,'DIRECT')

    def test_box_and_wall_and_cylinder_rejoin_before_arrival(self):
        for kind,width,depth in [('Box',2,2),('Wall',8,.6),('Cylinder',2,2)]:
            with self.subTest(shape=kind):
                d,m,s,c=world()
                m.add(kind,0,8,width,4,depth,d)
                c.go(V(0,2,18))
                # A visible object 7+ m away does not create an early detour.
                for _ in range(240):
                    c.tick(STEP)
                    self.assertEqual(c.navigator.state,'DIRECT')
                    self.assertAlmostEqual(d.p.x,0,places=5)
                self.fly(d,c)
                events=c.navigator.events
                self.assertEqual(events[0][0],'AVOID')
                self.assertGreater(events[0][1].z,4)
                self.assertTrue(any(state=='REJOIN' for state,_ in events))
                back=[p for state,p in events if state=='DIRECT']
                self.assertTrue(back)
                self.assertLess(abs(back[-1].x),.2)
                self.assertLess(back[-1].z,17)

    def test_obstacle_added_during_flight(self):
        d,m,s,c=world(); c.go(V(0,2,18))
        for _ in range(240): c.tick(STEP)
        self.assertEqual(c.navigator.state,'DIRECT')
        m.add('Box',0,7,2,4,2,d)
        self.fly(d,c)
        self.assertTrue(any(state=='AVOID' for state,_ in c.navigator.events))

    def test_two_obstacles_trigger_separate_avoidances(self):
        d,m,s,c=world()
        m.add('Box',0,7,2,4,2,d)
        m.add('Box',0,18,2,4,2,d)
        c.go(V(0,2,28))
        self.fly(d,c,12000)
        self.assertGreaterEqual(sum(state=='DIRECT' for state,_ in c.navigator.events),2)

    def test_avoidance_off_no_lateral_detour(self):
        d,m,s,c=world()
        m.add('Box',0,6,2,4,2,d); c.avoidance=False
        c.go(V(0,2,12))
        hit=False
        for _ in range(650):
            c.tick(STEP); hit=hit or c.collision
        self.assertTrue(hit)
        self.assertAlmostEqual(d.p.x,0,places=8)
        self.assertEqual(c.navigator.events,[])
        self.assertLessEqual(d.p.z,5-RADIUS)

    def test_high_speed_approach(self):
        d,m,s,c=world(); d.speed=5
        m.add('Box',0,10,2,4,2,d); c.go(V(0,2,22))
        self.fly(d,c)
        self.assertIn('AVOID',[state for state,_ in c.navigator.events])

    def test_sensor_only_memory_no_hidden_geometry(self):
        nav=ReactiveNavigator(); d=Drone(); d.p.y=2
        nav.begin(d.p,V(0,2,20))
        for _ in range(20): nav.command(d,SensorReading(),.1)
        self.assertEqual(nav.state,'DIRECT')
        self.assertEqual(nav.memory,{})
        obs=SensorReading(distance=1.5,sectors=dict(LEFT=1.,CENTER=1.5,RIGHT=4.),points=[(0,2,2.5)])
        nav.command(d,obs,.11)
        self.assertEqual(nav.state,'AVOID')
        self.assertEqual(nav.side,1)


class UIAndDepthTests(unittest.TestCase):
    def test_map_method_updates_focused_input_and_marker(self):
        import drone_sim_3d_2 as sim
        old=sim.u; sim.u=SimpleNamespace(camera=SimpleNamespace(ui=object()))
        try:
            ui=FlightUI.__new__(FlightUI)
            ui.wx=SimpleNamespace(text='',active=True); ui.wz=SimpleNamespace(text='',active=False)
            ui.fields=[ui.wx,ui.wz]; ui.message=SimpleNamespace(text='')
            mini=Minimap.__new__(Minimap); mini.size=.302
            mini.root=SimpleNamespace(get_position=lambda **kw:V(.613,.248,0))
            mini.pending_marker=SimpleNamespace(enabled=False,position=None)
            ui.minimap=mini; mini.on_pick=ui.map_pick
            self.assertTrue(mini.click(.613+.302*5/60,.248+.302*10/60))
            self.assertEqual((ui.wx.text,ui.wz.text),('5.0','10.0'))
            self.assertFalse(ui.wx.active); self.assertTrue(mini.pending_marker.enabled)
            self.assertFalse(mini.click(0,0))
        finally: sim.u=old

    def test_map_resize(self):
        for aspect in (1.6,16/9,21/9):
            cx=aspect/2-.187
            self.assertEqual(map_coordinates(cx,.248,cx,.248,.302),(0,0))
            self.assertIsNone(map_coordinates(cx+.4,.248,cx,.248,.302))

    def model_sensor(self,source='DEPTH',**changes):
        d,m,_,_=world()
        result=SimpleNamespace(position=d.p.tup(),yaw=0.,captured_at=monotonic(),valid=True,
            sectors=dict(LEFT=4.,CENTER=1.,RIGHT=3.),latency_ms=80.,points=((0,0,1.),))
        for k,v in changes.items(): setattr(result,k,v)
        worker=SimpleNamespace(error='',snapshot=lambda:result)
        return d,m,PerceptionSensor(worker,source)

    def test_depth_control_and_point_projection(self):
        d,m,s=self.model_sensor(); c=FlightController(d,m,s)
        reading=s.scan(d,m)
        self.assertAlmostEqual(reading.points[0][2],1.43)
        for _ in range(120): c.tick(STEP,1)
        self.assertLess(d.p.z,.17)
        self.assertEqual(s.reading.sector,'CENTER')

    def test_stale_invalid_wrong_pose_depth_hold(self):
        for changes in (dict(valid=False),dict(captured_at=monotonic()-5),dict(yaw=45),dict(position=(3,2,0))):
            d,m,s=self.model_sensor(**changes); c=FlightController(d,m,s)
            c.tick(STEP,1)
            self.assertTrue(s.hold); self.assertEqual(d.v.length(),0)

    def test_hybrid_fallback_explicit(self):
        d,m,s=self.model_sensor('HYBRID',valid=False)
        o=m.add('Box',0,4,2,4,2,d)
        reading=s.scan(d,m)
        self.assertIn('FALLBACK',s.note); self.assertFalse(s.hold)
        self.assertEqual(reading.obstacle_id,o.id)

    def test_metric_points_and_invalid_pixels(self):
        try: import numpy as np
        except ImportError: self.skipTest('NumPy unavailable')
        from depth_pipeline import summarize_depth,depth_points
        a=np.full((60,96),4.,dtype=np.float32)
        sectors,valid,heat=summarize_depth(a)
        self.assertTrue(valid); self.assertEqual(sectors['CENTER'],4.)
        points=depth_points(a)
        self.assertTrue(points)
        self.assertTrue(all(z==4 and abs(x)<3.1 for x,y,z in points))
        a[:]=float('nan')
        self.assertFalse(summarize_depth(a)[1]); self.assertEqual(depth_points(a),())


if __name__=='__main__': unittest.main(verbosity=2)
