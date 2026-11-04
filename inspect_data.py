import h5py

with h5py.File("data/ukdale/ukdale.h5", "r") as f:
    print("Top level:", list(f.keys()))
    elec = f["building1"]["elec"]
    print("Meters:", list(elec.keys())[:60])

    for m in ["meter1", "meter10", "meter12", "meter13"]:
        if m not in elec:
            print(m, "nahi mila")
            continue
        print("\n===", m, "===")
        print("Andar:", list(elec[m].keys()))
        t = elec[m]["table"]
        print("Shape:", t.shape)
        print("Dtype:", t.dtype)
        print("Pehli row:", t[0])
        print("Aakhri row:", t[-1])