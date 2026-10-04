"""Read-only MQTT collector. Requires a configured broker with device ACLs.
Never subscribes to command topics and never sends a pump command.
Network/broker operation must be commissioned on the team's actual network.
"""
from __future__ import annotations
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from .contracts import Telemetry
from .store import Ledger

LOG = logging.getLogger('chamgap.collector')
MAX_BYTES = 16384

def run() -> None:
    import paho.mqtt.client as mqtt
    host=os.environ['CHAMGAP_MQTT_HOST']
    user=os.environ['CHAMGAP_MQTT_USER']
    password=os.environ['CHAMGAP_MQTT_PASSWORD']
    port=int(os.getenv('CHAMGAP_MQTT_PORT','8883'))
    ca=os.getenv('CHAMGAP_MQTT_CA')
    insecure_lab=os.getenv('CHAMGAP_ALLOW_LOCAL_PLAINTEXT')=='1'
    if not ca and not insecure_lab:
        raise RuntimeError('Set TLS CA, or explicitly allow an isolated local lab network')
    db_path=Path(os.getenv('CHAMGAP_LEDGER','data/collector.sqlite3'))
    db_path.parent.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(str(db_path))
    client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,client_id='chamgap-readonly-v4')
    client.username_pw_set(user,password)
    if ca:
        client.tls_set(ca_certs=ca)
    client.reconnect_delay_set(min_delay=1,max_delay=30)

    def on_connect(c, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            LOG.error('broker connection rejected: %s',reason_code)
            return
        # Broker ACL must independently enforce which device may publish where.
        c.subscribe('chamgap/v4/+/+/telemetry',qos=1)

    def on_message(c,userdata,msg):
        try:
            if len(msg.payload)>MAX_BYTES:
                raise ValueError('oversized payload')
            t=Telemetry.model_validate_json(msg.payload)
            # Retained samples may initialize a display but cannot authorize a motor.
            inserted=ledger.ingest(t,msg.topic,datetime.now(timezone.utc))
            LOG.info('sample inserted=%s node=%s seq=%s retained=%s',
                     inserted,t.node_id,t.seq,msg.retain)
        except Exception as exc:
            # No credentials or entire payload are logged.
            LOG.warning('sample rejected type=%s reason=%s',type(exc).__name__,str(exc)[:240])

    client.on_connect=on_connect
    client.on_message=on_message
    client.connect(host,port,keepalive=30)
    try:
        client.loop_forever()
    finally:
        client.disconnect()
        ledger.close()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    run()
