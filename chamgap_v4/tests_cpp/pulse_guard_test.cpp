#include <cassert>
#include "../firmware_reference/include/pulse_guard.h"
int main(){
    PulseGuard g;
    assert(!g.active());
    assert(!g.start(0,1000,false));
    assert(!g.start(0,0,true));
    assert(!g.start(0,60001,true));
    assert(g.start(100,1000,true));
    assert(!g.start(101,1000,true));
    assert(g.tick(1099,true));
    assert(!g.tick(1100,true));
    assert(g.start(1200,500,true));
    assert(!g.tick(1201,false));
    assert(g.start(0xfffffff0u,32,true));
    assert(g.tick(0x0000000fu,true));
    assert(!g.tick(0x00000010u,true));
    assert(g.start(2000,200,true));
    g.stop(); assert(!g.active());
}
