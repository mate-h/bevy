import sys
sys.dont_write_bytecode = True
import array
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import argparse
import ibl_reference as ibl

class ReferenceTests(unittest.TestCase):
    def fixture(self, root, seed, spp, value=1):
        d=Path(root)/f'{seed}_{spp}'; d.mkdir()
        m=dict(schema_version=1,width=2,height=1,encoding='rgb32float-le',row_stride_bytes=24,
               actual_spp=spp,seed=seed,reference_file='reference.f32',geometry_file='geometry.mask',
               identity=dict(environment='furnace',view='front',roughness=.5,model='test'))
        (d/'manifest.json').write_text(json.dumps(m))
        a=array.array('f',[value]*6)
        if ibl.sys.byteorder!='little': a.byteswap()
        (d/'reference.f32').write_bytes(a.tobytes()); (d/'geometry.mask').write_bytes(bytes([1,1]))
        return ibl.read_capture(d)

    def test_disjoint_seeds_and_budgets(self):
        with tempfile.TemporaryDirectory() as t:
            caps=[self.fixture(t,s,8) for s in range(8)]
            caps += [self.fixture(t,s,4,.9) for s in range(4,8)]
            report=ibl.analyze(caps,set(range(4)),.05,.02)
            self.assertFalse(report['reference_qualified'])
            self.assertEqual(report['fixed_reference_spp'],8)
            self.assertEqual(report['budgets'][1]['seeds'],[4,5,6,7])
            self.assertEqual(report['budgets'][1]['uncertainty']['lit']['relative_rmse']['student_t_95_half_width'],0)
            self.assertLess(report['successive_budgets'][0]['lit_relative_rmse_change_mean'],0)

    def test_insufficient_and_missing(self):
        with tempfile.TemporaryDirectory() as t:
            caps=[self.fixture(t,0,8),self.fixture(t,1,8)]
            report=ibl.analyze(caps,{0})
            self.assertIsNone(report['budgets'][0]['uncertainty']['lit']['bias']['student_t_95_half_width'])
            self.assertEqual(ibl.analyze(caps,{7})['budgets'],[])

    def test_metadata_and_geometry_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.fixture(t,0,8); b=self.fixture(t,1,8)
            b['manifest']['view']='back'
            with self.assertRaisesRegex(ValueError,'view'): ibl.compatible([a,b])
            b['manifest']['view']='front'; b['geometry']=bytes([0,1])
            with self.assertRaisesRegex(ValueError,'geometry'): ibl.compatible([a,b])

    def test_reject_invalid_counts_and_radiance(self):
        with tempfile.TemporaryDirectory() as t:
            c=self.fixture(t,0,8); d=Path(c['directory'])
            original=json.loads((d/'manifest.json').read_text())
            for key,value in [('actual_spp',True),('seed',True),('actual_spp',16777217),('actual_spp',1.5)]:
                m={**original,key:value}
                (d/'manifest.json').write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError,'seed/spp'): ibl.read_capture(d)
            (d/'manifest.json').write_text(json.dumps(original))
            a=array.array('f',[-1]*6)
            (d/'reference.f32').write_bytes(a.tobytes())
            with self.assertRaisesRegex(ValueError,'radiance'): ibl.read_capture(d)

    def test_camera_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.fixture(t,0,8); b=self.fixture(t,1,8)
            a['manifest']['camera_transform']=[1,2,3]
            b['manifest']['camera_transform']=[2,2,3]
            with self.assertRaisesRegex(ValueError,'camera_transform'): ibl.compatible([a,b])

    def test_fixed_masks_and_black(self):
        ref=[([1,1,1],1),([0,0,0],2)]
        a=ibl.metrics([([2,2,2],1),([100,100,100],2)],ref,1)
        self.assertEqual(a['lit']['population'],1)
        self.assertAlmostEqual(a['lit']['bias'],1)
        self.assertIsNone(ibl.metrics([([1,1,1],1)],[([0,0,0],1)],0)['geometry']['relative_rmse'])

    def test_dry_run_and_mock_execution(self):
        with tempfile.TemporaryDirectory() as t:
            args=argparse.Namespace(output=t,binary='mock',cases=['furnace_front_50'],spp=[8],seeds=[0],filter='fis',execute=False,timeout=1)
            with patch('ibl_reference.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()) as stream:
                ibl.capture(args); run.assert_not_called()
            self.assertEqual(len(json.loads(stream.getvalue())),1)
            self.assertEqual(list(Path(t).iterdir()),[])
            def fake(command,env,**kw):
                self.assertNotIn('IBL_PROFILE',env)
                self.assertEqual(kw['timeout'],1)
                self.assertTrue(Path(command[0]).is_absolute())
                self.assertEqual(env['BEVY_ASSET_ROOT'],str(Path(ibl.__file__).resolve().parent.parent))
                self.fixture(env['IBL_CAPTURE_DIR'],int(env['IBL_REFERENCE_SEED']),int(env['IBL_REFERENCE_SPP']))
            args.execute=True
            with patch.dict(ibl.os.environ, {'IBL_PROFILE':'1'}), patch('ibl_reference.subprocess.run',side_effect=fake): ibl.capture(args)
            with self.assertRaisesRegex(ValueError,'already exists'): ibl.capture(args)

if __name__=='__main__': unittest.main()
