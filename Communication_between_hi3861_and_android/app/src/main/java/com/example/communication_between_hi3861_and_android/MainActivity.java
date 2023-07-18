package com.example.communication_between_hi3861_and_android;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.text.TextUtils;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
    private MyTcpClient myTcpClient;
    private SeekBar mSeekBar;
    private TextView mTextView;
    private TextView tv_lux;
    private EditText ip_et;
    private EditText port_et;
    private Button connect_btn;
    private Handler handler;
    private int connect_flag;//连接与断开的标志
    int luxtemp = 0;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mSeekBar = findViewById(R.id.seek_bar);
        mTextView = findViewById(R.id.tv_progress);
        tv_lux = findViewById(R.id.tv_lux);
        ip_et = findViewById(R.id.ip_et);
        port_et = findViewById(R.id.port_et);
        connect_btn = findViewById(R.id.connect_btn);
        mSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override  //当滑块进度改变时，会执行该方法下的代码
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                mTextView.setText("当前亮度： " + i + "/90");
                char str = (char) (i + 32);
                String send = String.valueOf(str);
                send_Cmd(send);
            }

            @Override  //当开始滑动滑块时，会执行该方法下的代码
            public void onStartTrackingTouch(SeekBar seekBar) {
                Toast.makeText(MainActivity.this, "开始调节亮度", Toast.LENGTH_SHORT).show();
//                send_Cmd("A");
            }

            @Override   //当结束滑动滑块时，会执行该方法下的代码
            public void onStopTrackingTouch(SeekBar seekBar) {
                Toast.makeText(MainActivity.this, "结束调节亮度", Toast.LENGTH_SHORT).show();
//                send_Cmd("a");
            }
        });
        //每秒更新一次textview
        Thread t = new Thread() {
            @Override
            public void run() {
                try {
                    while (!isInterrupted()) {
                        Thread.sleep(100);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                String luxtem = Integer.toString(luxtemp);
                                tv_lux.setText("当前光照强度：" + luxtem);
                            }
                        });
                    }
                } catch (InterruptedException e) {
                }
            }
        };
        t.start();
        handler = new Handler(Looper.getMainLooper()) {
            @Override
            public void handleMessage(@NonNull Message msg) {
                super.handleMessage(msg);
                switch (msg.what) {
                    case 1:    //连接成功
                        connect_flag = 1;
                        rcvMsgHandler();
                        connect_btn.setText("断开连接");
                        Toast.makeText(MainActivity.this, "连接成功", Toast.LENGTH_SHORT).show();
                        break;
                    case 2:    //连接失败
                        connect_flag = 0;
                        Toast.makeText(MainActivity.this, "连接失败", Toast.LENGTH_SHORT).show();
                        break;
                    case 3:     //断开连接
                        connect_flag = 0;
                        connect_btn.setText("连接");
                        Toast.makeText(MainActivity.this, "连接已断开", Toast.LENGTH_SHORT).show();
                        break;
                }
            }
        };
        final Button btn1 = (Button) findViewById(R.id.anticlockwise_btn);
        btn1.setOnTouchListener((v, event) -> {
            if (v.getId() == R.id.anticlockwise_btn) {
                if (event.getAction() == MotionEvent.ACTION_DOWN) {
                    Toast.makeText(MainActivity.this, "电机开始逆时针旋转", Toast.LENGTH_SHORT).show();
                    send_Cmd("{");
                } else if (event.getAction() == MotionEvent.ACTION_UP) {
                    Toast.makeText(MainActivity.this, "电机结束逆时针旋转", Toast.LENGTH_SHORT).show();
                    send_Cmd("|");
                }
            }
            return false;
        });
        final Button btn2 = (Button) findViewById(R.id.clockwise_btn);
        btn2.setOnTouchListener((v, event) -> {
            if (v.getId() == R.id.clockwise_btn) {
                if (event.getAction() == MotionEvent.ACTION_DOWN) {
                    Toast.makeText(MainActivity.this, "电机开始顺时针旋转", Toast.LENGTH_SHORT).show();
                    send_Cmd("}");
                } else if (event.getAction() == MotionEvent.ACTION_UP) {
                    Toast.makeText(MainActivity.this, "电机结束顺时针旋转", Toast.LENGTH_SHORT).show();
                    send_Cmd("~");
                }
            }
            return false;
        });
    }

    public void rcvMsgHandler() {
        new Thread(() -> {
            while (true) {
                if (myTcpClient.serverStatus != -1) {
                    String Msg = myTcpClient.rcvMsg();
                    if (Msg != null) {
                        Log.v("接收到服务端的消息", Msg);
                        int lux = Integer.parseInt(Msg);
                        luxtemp = lux;
                    } else {
                        Log.v("rcvMsgHandler", "内存已释放");
                        break;
                    }
                } else {
                    break;
                }
            }
        }).start();
    }

    public void send_Cmd(String msg) {
        if (connect_flag == 1) {
            new Thread(() -> {
                myTcpClient.sendMsg(msg);
            }).start();
        }
    }

    public void Connect(View v) {
        String IP = ip_et.getText().toString().trim();
        String PORT = port_et.getText().toString();
        int Port;

        if (connect_flag == 0) {
            myTcpClient = new MyTcpClient();
            if (TextUtils.isEmpty(IP)) {
                Toast.makeText(MainActivity.this, "请输入IP", Toast.LENGTH_SHORT).show();
                ip_et.requestFocus();
            } else if (TextUtils.isEmpty(PORT)) {
                Toast.makeText(MainActivity.this, "请输入端口", Toast.LENGTH_SHORT).show();
                port_et.requestFocus();
            } else {
                Port = Integer.parseInt(PORT);
                new Thread(() -> {
                    if (myTcpClient.connect(IP, Port)) {
                        Message msg = new Message();
                        msg.what = 1;
                        handler.sendMessage(msg);
                    } else {
                        Message msg = new Message();
                        msg.what = 2;
                        handler.sendMessage(msg);
                    }
                }).start();
            }
        } else {
            myTcpClient.disConnect();
            Message msg = new Message();
            msg.what = 3;
            handler.sendMessage(msg);
        }
    }

    public void voidFunction(View v) {
    }
}