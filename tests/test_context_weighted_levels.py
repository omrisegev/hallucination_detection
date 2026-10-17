import unittest
import numpy as np
from spectral_utils.context_weighted_levels import held_weights,selected_tokens,readout,decompose,score_answer
from spectral_utils.residual_moment_fusion import stream_top10

class ContextLevelTests(unittest.TestCase):
    def test_static_and_negative_readout(self):
        x=np.random.default_rng(1).normal(size=(80,5));spans=[[0,30],[30,80]];w=np.array([-.3,.2,.1,.3,.1])
        np.testing.assert_allclose(readout(x,selected_tokens(x,spans),w),stream_top10(x,spans)@w,atol=1e-14)
    def test_no_future_anchor_and_warmup(self):
        pos=[16,32];w=np.array([[1.,0],[0,1.]])
        out=held_weights(40,pos,w,[.5,.5]);changed=held_weights(40,pos,w+[0,100],[.5,.5])
        np.testing.assert_array_equal(out[:16],np.full((16,2),.5))
        np.testing.assert_array_equal(out[:16],changed[:16])
        w[1]=[12,14];np.testing.assert_array_equal(held_weights(40,pos,w,[.5,.5])[:32],out[:32])
    def test_tie_break(self):
        ids=selected_tokens(np.ones((20,3)),[[0,20]])[0]
        np.testing.assert_array_equal(ids,np.tile(np.arange(10,20)[:,None],(1,3)))
    def test_amplitude_direction(self):
        static=np.array([.2,.3,.5]);w=np.array([[.4,.6,1.],[.1,.7,.2]]);sd=np.array([2,3,5])
        amp,direction,A=decompose(w,static,sd)
        np.testing.assert_allclose(direction*A[:,None],w)
        np.testing.assert_allclose(np.linalg.norm(direction*sd,axis=1),np.linalg.norm(static*sd))
        np.testing.assert_allclose(amp[0],w[0])
    def test_switching_expert_mechanics(self):
        # Correct weights are PROVIDED; this does not validate their estimation.
        x=np.zeros((48,2));x[20]=[3,10];x[36]=[10,4]
        w=held_weights(48,[16,32],[[1,0],[0,1]],[.5,.5])
        np.testing.assert_array_equal(w[20],[1,0]);np.testing.assert_array_equal(w[36],[0,1])
        selected=selected_tokens(x,[[16,32],[32,48]])
        np.testing.assert_allclose(readout(x,selected,w),[.3,.4])
        np.testing.assert_allclose(readout(x,selected,[.5,.5]),[.65,.7])
    def test_constant_context_identity(self):
        x=np.random.default_rng(2).normal(size=(50,5));heads={h:np.ones(5)/5 for h in ('native','simplex','group')}
        local={h:{a:np.tile(w,(2,1)) for a in ('energy','position','random')} for h,w in heads.items()}
        out=score_answer(x,[[0,20],[20,50]],[16,30],local,heads,np.ones(5))
        for v in out.values():np.testing.assert_allclose(v,out['equal'],atol=1e-14)

if __name__=='__main__':unittest.main()
