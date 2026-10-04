from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from chamgap.contracts import Action, FEATURES, F, action_mask
from chamgap.simulator import SmartFarmCore
from chamgap.connectome import random_mask, rewire, save_mask, load_mask
from chamgap.model import ValueModel, choose_action
from chamgap.safety import DryRunExecutor, CheckRequest


def test_reset_deterministic():
    a=SmartFarmCore(); b=SmartFarmCore()
    x,_=a.reset(seed=23); y,_=b.reset(seed=23)
    assert np.array_equal(x,y)


def test_clone_reproduces_same_action():
    e=SmartFarmCore(); e.reset(seed=24)
    a=e.clone(6); b=e.clone(6)
    xa,ra,ta,ua,_=a.step(0); xb,rb,tb,ub,_=b.step(0)
    assert np.array_equal(xa,xb) and (ra,ta,ua)==(rb,tb,ub)


def test_no_truth_field_in_schema():
    assert not any('fault' in s or 'true_' in s or 'scenario' in s for s in FEATURES)


def test_non_soil_cannot_water():
    for channel in (1,2):
        e=SmartFarmCore(); x,_=e.reset(seed=9,options={'channel':channel})
        assert not action_mask(x)[2]
        with pytest.raises(ValueError): e.step(2)

@pytest.mark.parametrize('key',['estop','leak'])
def test_interlock_blocks_pulse(key):
    e=SmartFarmCore(); x,_=e.reset(seed=9,options={'channel':0,key:True})
    assert not action_mask(x)[2]


def test_unseen_comparison_not_in_obs():
    e=SmartFarmCore(); e.reset(seed=7)
    e.peer_valid=False; e.peer=999.
    x=e._obs()
    assert x[F['comparison_scaled']]==0 and x[F['pair_abs_difference']]==0


def test_mask_uses_observation_not_truth():
    e=SmartFarmCore(); x,_=e.reset(seed=7,options={'channel':0})
    m=action_mask(x)
    e.scenario='supply'; e.blockage=1.; e.true_water=0.
    assert np.array_equal(m,action_mask(x))


def test_inspection_is_abstention_not_success():
    e=SmartFarmCore(); e.reset(seed=7)
    _,r,term,trunc,info=e.step(3)
    assert term and not trunc and r<0
    assert info['diagnosis']=='UNKNOWN' and info['human_request']


def test_rewiring_preserves_both_degrees():
    m=random_mask(); z,meta=rewire(m,8)
    assert np.array_equal(m.sum(0),z.sum(0))
    assert np.array_equal(m.sum(1),z.sum(1))
    assert meta['successful_swaps']>0
    assert not np.array_equal(m,z)


def test_demo_cannot_be_called_real(tmp_path):
    p=tmp_path/'mask.npy'; save_mask(p,random_mask(),'DEMO_RANDOM_NOT_BIOLOGICAL')
    with pytest.raises(ValueError): load_mask(p,require_real=True)


def test_gradient_reaches_adapter_but_not_topology():
    torch.manual_seed(7)
    m=ValueModel('fly',random_mask())
    x=torch.randn(16,len(FEATURES))
    y=m(x); y.square().mean().backward()
    assert m.adapter.weight.grad is not None
    assert m.adapter.weight.grad.abs().sum()>0
    assert m.adjacency.requires_grad is False
    assert 'adjacency' not in dict(m.named_parameters())


def test_model_choice_honors_mask():
    m=ValueModel('mlp')
    e=SmartFarmCore(); x,_=e.reset(seed=7)
    a,_=choose_action(m,x,np.array([False,False,False,True]))
    assert a==3


def test_request_expiry_and_duplicate(tmp_path):
    db=str(tmp_path/'ledger.sqlite')
    e=SmartFarmCore(); x,_=e.reset(seed=7)
    executor=DryRunExecutor(db)
    req=CheckRequest('one',0,110.)
    assert executor.accept(req,x,now=100.)['status']=='DRY_RUN_ACCEPTED'
    executor.db.close()
    again=DryRunExecutor(db)
    assert again.accept(req,x,now=101.)['status']=='DUPLICATE'
    assert again.accept(CheckRequest('two',0,90.),x,now=100.)['reason']=='expired'


def test_mass_stays_within_bounds():
    e=SmartFarmCore(); e.reset(seed=8)
    e._advance(3600.,10.)
    assert 0<=e.true_water<=e.theta_sat*e.soil_volume


def test_remaining_budget_blocks_pulse():
    e=SmartFarmCore(); x,_=e.reset(seed=7,options={'channel':0})
    x[F['remaining_water_scaled']]=0.
    assert not action_mask(x)[2]


def test_reject_invalid_observation():
    from chamgap.contracts import check_observation
    with pytest.raises(ValueError): check_observation(np.zeros(1))
    with pytest.raises(ValueError): check_observation(np.full(len(FEATURES),np.nan))


def test_real_feature_builder_rejects_future():
    from chamgap.features import Evidence,Reading,build_observation
    e=Evidence(channel='soil',target_window=(Reading(101.,.4),))
    with pytest.raises(ValueError): build_observation(e,100.)


def test_real_feature_defaults_fail_closed():
    from chamgap.features import Evidence,Reading,build_observation
    e=Evidence(channel='soil',target_window=(Reading(99.,.4),Reading(100.,.4)))
    x=build_observation(e,100.)
    assert x.shape==(len(FEATURES),) and not action_mask(x)[2]
