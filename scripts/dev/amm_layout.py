"""Empirically pin down PumpSwap Buy/SellEvent field semantics from raw blocks."""
import gzip, json, hashlib, struct, sys, collections
import base58
PAMM="pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"; TAG=bytes.fromhex("e445a52e51cb9a1d")
d=lambda n: hashlib.sha256(f"event:{n}".encode()).digest()[:8]
BUY,SELL=d("BuyEvent"),d("SellEvent")
rows=[]
for i,line in enumerate(gzip.open(sys.argv[1],"rt")):
    if i>=300: break
    r=json.loads(line)
    for tx in r.get("txs",[]):
        m=tx["meta"]
        if m["err"]: continue
        la=m.get("loadedAddresses") or {}
        keys=tx["transaction"]["message"]["accountKeys"]+la.get("writable",[])+la.get("readonly",[])
        for g in m.get("innerInstructions") or []:
            for ins in g["instructions"]:
                if keys[ins["programIdIndex"]]!=PAMM: continue
                b=base58.b58decode(ins["data"])
                if b[:8]!=TAG or b[8:16] not in (BUY,SELL): continue
                x=b[16:]; u=struct.unpack_from("<q13Q",x,0)
                pool=base58.b58encode(x[112:144]).decode()
                rows.append((r["slot"],tx["idx"],b[8:16]==BUY,pool,u))
byp=collections.defaultdict(list)
for row in rows: byp[row[3]].append(row)
names=["ts","base_amt","limit","ub","uq","pool_base","pool_quote","q_amt","lp_bps","lp_fee","pr_bps","pr_fee","q2","user_q"]
print("example buy:",dict(zip(names,[r for r in rows if r[2]][0][4])))
print("example sell:",dict(zip(names,[r for r in rows if not r[2]][0][4])))
# test hypotheses for pool quote change between consecutive events
hyp=collections.Counter(); n=0
for p,lst in byp.items():
    for a,b in zip(lst,lst[1:]):
        dq=b[4][6]-a[4][6]; db=b[4][5]-a[4][5]; u=dict(zip(names,a[4])); n+=1
        s=1 if a[2] else -1
        for hname,val in [("q_amt",u["q_amt"]),("q2",u["q2"]),("user_q",u["user_q"]),("q_amt+lp",u["q_amt"]+u["lp_fee"]),("q_amt-lp",u["q_amt"]-u["lp_fee"]),("q2-lp",u["q2"]-u["lp_fee"])]:
            if dq==s*val: hyp[(a[2],hname)]+=1
        if db==-s*u["base_amt"]: hyp[(a[2],"base_ok")]+=1
print(n, sorted(hyp.items()))
