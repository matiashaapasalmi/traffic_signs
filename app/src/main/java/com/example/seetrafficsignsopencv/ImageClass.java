package com.example.seetrafficsignsopencv;

public enum ImageClass {
    EMPTY (R.drawable.sl60),
    SL_30 (R.drawable.sl60),
    SL_40 (R.drawable.sl120),
    SL_50 (R.drawable.sl120),
    SL_60 (R.drawable.sl30),
    SL_70 (R.drawable.sl30),
    SL_80 (R.drawable.sl60),
    SL_100 (R.drawable.sl60),
    SL_120 (R.drawable.sl120);

    private final int id;

    ImageClass(int picID) {
        this.id = picID;
    }
    public int id() { return id; }
}
