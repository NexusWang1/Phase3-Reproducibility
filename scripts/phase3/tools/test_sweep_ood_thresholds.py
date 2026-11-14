"""CPU-only consistency tests for the offline AP/veto sweep."""
import unittest
from sweep_ood_thresholds import ap,map50,sweep,operating_points


def row(image,dataset,cls,conf,energy,maha):
    return {'image':image,'dataset':dataset,'cls':cls,'conf':conf,
            'box_xyxy':[0.,0.,10.,10.],'energy':energy,'maha':maha}


class SweepTests(unittest.TestCase):
    def test_ap_image_isolation(self):
        truth=[row('a','rgbt_val',0,1.,0.,0.)]
        wrong=[row('b','rgbt_val',0,.9,0.,0.)]
        self.assertEqual(ap(wrong,truth),0.)
        self.assertAlmostEqual(ap([wrong[0],row('a','rgbt_val',0,.8,0.,0.)],truth),.5)

    def test_person_exempt_and_threshold_identity(self):
        truth=[row('a','rgbt_val',0,1.,0.,0.)]
        scores=[row('a','rgbt_val',0,.2,1.,1.),row('h','m3fd_hardneg',0,.1,2.,2.),
                row('h','m3fd_hardneg',2,.01,3.,3.)]
        baseline=map50(scores[:1],truth)
        points=sweep(scores,truth,'conf',[.045,.15],baseline)
        self.assertEqual(points[0]['hardneg_fp'],1)
        self.assertEqual(points[1]['hardneg_fp'],0)
        self.assertEqual(points[1]['hardneg_person_fp'],1)
        self.assertEqual(points[0]['map50'],baseline)
        self.assertEqual(operating_points(points)['map_lossless']['hardneg_fp'],0)

    def test_null_is_kept(self):
        truth=[row('a','rgbt_val',0,1.,0.,0.)]
        scores=[row('a','rgbt_val',0,.2,None,1.),row('h','m3fd_hardneg',0,.1,None,2.)]
        point=sweep(scores,truth,'energy',[0.],map50(scores[:1],truth))[0]
        self.assertEqual(point['hardneg_fp'],1)
        self.assertEqual(point['null_event_scores_kept'],2)


if __name__=='__main__':unittest.main()
