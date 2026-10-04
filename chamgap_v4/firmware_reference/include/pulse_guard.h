#pragma once
#include <cstdint>
// Pure logic reference, no GPIO. NOT a complete hardware safety implementation.
// Persistent command deduplication and budget reservation must precede start().
class PulseGuard {
    bool active_=false;
    std::uint32_t began_=0, duration_=0;
public:
    bool start(std::uint32_t now,std::uint32_t duration,bool all_interlocks){
        if(active_ || !all_interlocks || duration==0 || duration>60000) return false;
        began_=now; duration_=duration; active_=true; return true;
    }
    bool tick(std::uint32_t now,bool all_interlocks){
        if(active_ && (!all_interlocks || std::uint32_t(now-began_)>=duration_)) active_=false;
        return active_; // true requests ON; physical cutoff remains independent
    }
    void stop(){active_=false;}
    bool active() const{return active_;}
};
