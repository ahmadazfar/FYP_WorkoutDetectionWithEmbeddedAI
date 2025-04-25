#include "MPU6050test.h"
#include <stdint.h>
#include <stdlib.h>
#include "mxc.h"
#include "board.h"

static mxc_i2c_req_t i2c_req;


static inline int reg_write(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = { reg, val };

    i2c_req.i2c = MXC_I2C1;
    i2c_req.addr = 0x68;
    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_len = 0;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

static inline int reg_read(uint8_t reg, uint8_t *dat)
{
    uint8_t buf[1] = { reg };

    i2c_req.i2c = MXC_I2C1;
    i2c_req.addr = 0x68;
    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = dat;
    i2c_req.rx_len = 1;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

MPU6050_t mpu6050;

void MPU6050(uint8_t address) {

    mpu6050.devAddr = address;
}


void MPU6050_initialize() {
	uint8_t temp_data = MPU6050_GYRO_FS_500;
	reg_write(MPU6050_RA_GYRO_CONFIG, &temp_data);
	uint8_t temp_data1 = MPU6050_ACCEL_FS_8;
	reg_write(MPU6050_RA_ACCEL_CONFIG, &temp_data1);
	reg_write(MPU6050_RA_PWR_MGMT_1, 0);
}


bool MPU6050_testConnection() {
    return MPU6050_getDeviceID() == 0x75;
}

void MPU6050_ReadAccel(mxc_i2c_regs_t *i2c, int16_t *accel_x, int16_t *accel_y, int16_t *accel_z) {
    // MPU6050 device address

    uint8_t buffer[6];
    unsigned int len = 6;
    // Read accelerometer data from MPU6050
    int status = MXC_I2C_Read(&i2c,&buffer[0], &len, 1); // Pass the address of the buffer

        if (status == 0) {
            // Extract accelerometer data from the buffer
            *accel_x = (buffer[0] << 8) | buffer[1];
            *accel_y = (buffer[2] << 8) | buffer[3];
            *accel_z = (buffer[4] << 8) | buffer[5];
        } else {
            // Handle error condition
            // You can add error handling code here
        }
}


int mpu6050_get_acc_data(int16_t *ptr)
{
    uint8_t buf[1] = { MPU6050_RA_ACCEL_XOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 6;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

int mpu6050_get_gyro_data(int16_t *ptr)
{
    uint8_t buf[1] = { MPU6050_RA_GYRO_XOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 6;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

int mpu6050_get_z_data(int16_t *ptr)
{
    uint8_t buf[1] = { MPU6050_RA_ACCEL_ZOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 2;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

int MPU6050_READ_X_ACC(int16_t *ptr){
    uint8_t buf[1] = { MPU6050_RA_ACCEL_XOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 2;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

int MPU6050_READ_Y_ACC(int16_t *ptr){
    uint8_t buf[1] = { MPU6050_RA_ACCEL_YOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 2;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

int MPU6050_READ_Z_ACC(int16_t *ptr){
    uint8_t buf[1] = { MPU6050_RA_ACCEL_ZOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 2;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

int MPU6050_READ_X_GYRO(int16_t *ptr){
    uint8_t buf[1] = { MPU6050_RA_GYRO_XOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 2;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

int MPU6050_READ_Y_GYRO(int16_t *ptr){
    uint8_t buf[1] = { MPU6050_RA_GYRO_YOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 2;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

int MPU6050_READ_Z_GYRO(int16_t *ptr){
    uint8_t buf[1] = { MPU6050_RA_GYRO_ZOUT_H };

    i2c_req.tx_buf = buf;
    i2c_req.tx_len = sizeof(buf);
    i2c_req.rx_buf = (uint8_t *)ptr;
    i2c_req.rx_len = 2;

    return MXC_I2C_MasterTransaction(&i2c_req);
}

bool I2Cdev_writeBit(uint8_t regAddr, uint8_t bitNum, uint8_t data) {
    uint8_t b;
    reg_read(regAddr, &b);
    b = (data != 0) ? (b | (1 << bitNum)) : (b & ~(1 << bitNum));
    return reg_read(regAddr, &b);
}

bool I2Cdev_writeBits( uint8_t regAddr, uint8_t bitStart, uint8_t length, uint8_t data) {
    //      010 value to write
    // 76543210 bit numbers
    //    xxx   args: bitStart=4, length=3
    // 00011100 mask byte
    // 10101111 original value (sample)
    // 10100011 original & ~mask
    // 10101011 masked | value
    uint8_t b;

    if ( reg_read(regAddr, &b)) {

        uint8_t mask = ((1 << length) - 1) << (bitStart - length + 1);
        data <<= (bitStart - length + 1); // shift data into correct position
        data &= mask; // zero all non-important bits in data
        b &= ~(mask); // zero all important bits in existing byte
        b |= data; // combine data with existing byte
        return reg_write(regAddr, b);
    } else {
        return false;
    }
}

uint8_t MPU6050_getDeviceID() {
    I2Cdev_readBits(MPU6050_RA_WHO_AM_I, MPU6050_WHO_AM_I_BIT, MPU6050_WHO_AM_I_LENGTH, *mpu6050.buffer);
    return mpu6050.buffer[1];
}

int8_t I2Cdev_readBits(uint8_t regAddr, uint8_t bitStart, uint8_t length, uint8_t *data) {
    // 01101001 read byte
    // 76543210 bit numbers
    //    xxx   args: bitStart=4, length=3
    //    010   masked
    //   -> 010 shifted
    uint8_t count, b;
    reg_read(MPU6050_RA_XG_OFFS_USRH, &b);
    printf("%d\n",b);
    if ((count = reg_read(regAddr, &b))) {
        uint8_t mask = ((1 << length) - 1) << (bitStart - length + 1);
        b &= mask;
        b >>= (bitStart - length + 1);
        *data = b;
    }
    return count;
}
