package com.example.communication_between_hi3861_and_android;

import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;

public class MyTcpClient {
    private Socket client;
    private OutputStream out;
    private InputStream in;
    public int serverStatus = 1;

    public MyTcpClient() {
    }

    public boolean connect(String IP, int Port) {
        boolean isConnect = false;
        try {
            if (client == null) {
                client = new Socket();
            }
            SocketAddress socketAddress = new InetSocketAddress(IP, Port);//设置IP和端口
            client.connect(socketAddress, 2000);//连接
            if (client.isConnected()) {
                Log.v("MyTcpClient------>", "成功连接服务器");
                isConnect = true;
            }
        } catch (IOException e) {
            Log.v("MyTcpClient------>", "连接服务器失败" + e.getMessage());
            isConnect = false;
            e.printStackTrace();
        }
        return isConnect;
    }

    //发送数据和信息
    public void sendMsg(String msg) {
        try {
            if (out == null) {
                out = client.getOutputStream();
            }
            out.write(msg.getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String rcvMsg() {
        byte[] msg = new byte[128];
        int msg_len;
        String Msg = null;
        try {
            if (in == null) {
                in = client.getInputStream();
            }
            msg_len = in.read(msg);
            if (msg_len == -1) {
                serverStatus = msg_len;
                Log.v("MyTcpClient------>", "服务器已断开");
                closeAll();
                return null;
            }
            Msg = new String(msg, 0, msg_len);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return Msg;
    }

    //关闭所有
    public void closeAll() {
        try {
            if (out != null) {
                out.close();
                out = null;
            }
            if (in != null) {
                in.close();
                in = null;
            }
            if (client != null) {
                client.close();
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    //断开连接
    public void disConnect() {
        closeAll();
    }
}

